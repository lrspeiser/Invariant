"""One frozen open-gravity development campaign and its terminal session reservation.

The preflight half of this module is strictly zero-response.  It binds the complete
live mechanism-card set, the source-availability matrix, one immutable manifest,
and the append-only ledger that permanently reserves the session's only scored
campaign.  Production response loading is reachable only after those artifacts
are committed and validated byte-for-byte.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_registry_foundation_v1 as registry
from sigma_theory_compiler import open_gravity_source_availability_contract_v2 as sources
from sigma_theory_compiler import open_gravity_static_radial_adapter_v1 as adapter
from sigma_theory_compiler import twell_400_v2_typed_compiler_final_v3 as twell

CONFIG_PATH = Path("configs/open_gravity_campaign_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_campaign_v1.py")
TEST_PATH = Path("tests/test_open_gravity_campaign_v1.py")
MANIFEST_PATH = Path("runs/gravity/open-gravity-campaign-v1/manifest.json")
TERMINAL_LEDGER_PATH = Path("runs/gravity/open-gravity-campaign-v1/terminal-ledger.json")
PREFLIGHT_PATH = Path("runs/gravity/open-gravity-campaign-v1/preflight.json")
ACCESS_INTENT_PATH = Path("runs/gravity/open-gravity-campaign-v1/access-intent.json")
RESULT_PATH = Path("runs/gravity/open-gravity-campaign-v1/result.json")
ADJUDICATION_PATH = Path("runs/gravity/open-gravity-campaign-v1/adjudication.json")
FAILURE_PATH = Path("runs/gravity/open-gravity-campaign-v1/failure.json")
ARTIFACT_DIRECTORY = Path("runs/gravity/open-gravity-campaign-v1/artifacts")

CAMPAIGN_SCHEMA = "invariant-open-gravity-campaign-manifest-1.0"
TERMINAL_SCHEMA = "invariant-open-gravity-terminal-campaign-ledger-1.0"
PREFLIGHT_SCHEMA = "invariant-open-gravity-campaign-preflight-1.0"
ACCESS_INTENT_SCHEMA = "invariant-open-gravity-campaign-access-intent-1.0"
RESULT_SCHEMA = "invariant-open-gravity-campaign-result-1.0"
ADJUDICATION_SCHEMA = "invariant-open-gravity-campaign-adjudication-1.0"
FAILURE_SCHEMA = "invariant-open-gravity-campaign-failure-1.0"

MANDATORY_CAMPAIGN_ARTIFACTS = (
    "global-cell-ledger.json",
    "counterexample-ledger.json",
    "closure-matrix.json",
    "blocked-idea-ledger.json",
    "comparator-ledger.json",
    "multiplicity-ledger.json",
    "matched-environment-discriminator.json",
    "repair-ledger.json",
    "lay-summary.json",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_CONFIG_CONTENT_SHA256 = "4a830275205fbc0d792dfe3b8abd723fc5b272447d598b6afe0be2b2bc93e9a1"
EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256 = "053aed14ea8a4ca758cd2cb748ee4e41c1f6560cba71bacc2d23658805155362"  # fmt: skip
EXPECTED_TEST_FILE_SHA256 = "5f6cd7b1cdc983d93874f0ba89904b4f3e140f39ca3e8db218405b92d15112c3"
EXPECTED_TEST_SEMANTIC_SHA256 = "5f6cd7b1cdc983d93874f0ba89904b4f3e140f39ca3e8db218405b92d15112c3"
_IMPLEMENTATION_PIN_NAME = "EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256"
_IMPLEMENTATION_PIN_RE = re.compile(
    rf'^{_IMPLEMENTATION_PIN_NAME} = "[^"]+"(?:  # fmt: skip)?$', re.MULTILINE
)

DOMAINS = ("GALAXIES", "GROUPS", "CLUSTERS", "LENSING")
ZERO_ACCESS = {
    "scientific_response_files_opened": 0,
    "scientific_response_rows_opened": 0,
    "development_response_rows_opened": 0,
    "group_response_rows_opened": 0,
    "lensing_response_rows_opened": 0,
    "confirmation_rows_opened": 0,
    "independent_rows_opened": 0,
    "scientific_scores_computed": 0,
    "network_calls": 0,
    "model_calls": 0,
    "paid_calls": 0,
}


class OpenGravityCampaignError(RuntimeError):
    """Fail-closed campaign contract error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenGravityCampaignError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_semantic_sha256(payload: bytes) -> str:
    normalized = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _implementation_semantic_sha256(payload: bytes) -> str:
    normalized = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_IMPLEMENTATION_PIN_RE.finditer(normalized))
    _require(len(matches) == 1, "implementation semantic pin assignment changed")
    normalized = _IMPLEMENTATION_PIN_RE.sub(
        f'{_IMPLEMENTATION_PIN_NAME} = "<SELF_PIN>"  # fmt: skip', normalized
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_local_artifact_integrity() -> None:
    root = _repo_root()
    config = _read_json(root / CONFIG_PATH)
    _require(
        content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256,
        "campaign config semantic seal changed",
    )
    _require(
        _implementation_semantic_sha256((root / MODULE_PATH).read_bytes())
        == EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256,
        "campaign implementation semantic seal changed",
    )
    test_bytes = (root / TEST_PATH).read_bytes()
    _require(
        hashlib.sha256(test_bytes).hexdigest() == EXPECTED_TEST_FILE_SHA256,
        "campaign test raw seal changed",
    )
    _require(
        _text_semantic_sha256(test_bytes) == EXPECTED_TEST_SEMANTIC_SHA256,
        "campaign test semantic seal changed",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = copy.deepcopy(dict(value))
    body.pop(field, None)
    return content_sha256(body)


def _exact_paths(config: Mapping[str, Any]) -> None:
    expected = {
        "manifest": MANIFEST_PATH.as_posix(),
        "terminal_ledger": TERMINAL_LEDGER_PATH.as_posix(),
        "preflight": PREFLIGHT_PATH.as_posix(),
        "access_intent": ACCESS_INTENT_PATH.as_posix(),
        "result": RESULT_PATH.as_posix(),
        "adjudication": ADJUDICATION_PATH.as_posix(),
        "failure": FAILURE_PATH.as_posix(),
        "artifact_directory": ARTIFACT_DIRECTORY.as_posix(),
    }
    _require(config.get("output_paths") == expected, "campaign output paths changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256,
        "campaign config semantic seal changed",
    )
    _require(
        config.get("schema_version") == "invariant-open-gravity-campaign-config-1.0",
        "campaign config schema changed",
    )
    _require(config.get("campaign_id") == "OPEN-GRAVITY-CAMPAIGN-v1", "campaign ID changed")
    _require(
        config.get("session_id") == registry.TRUSTED_SESSION_ID,
        "trusted session ID changed",
    )
    _require(config.get("semantic_version") == "1.0.0", "campaign version changed")
    _require(
        config.get("status")
        == "FROZEN_UNRUN_ZERO_RESPONSE_ROW_ACCESS_ONE_DISCLOSED_OPAQUE_HASH_READ",
        "campaign status overclaims execution",
    )
    authority = config.get("authority", {})
    _require(authority.get("standing_user_authorization") is True, "standing scope missing")
    _require(authority.get("campaign_count") == 1, "campaign count changed")
    _require(authority.get("automatic_second_campaign") is False, "second campaign enabled")
    _require(float(authority.get("external_cost_usd", -1.0)) == 0.0, "external cost changed")
    goal = config.get("goal_bindings", {})
    _require(
        goal.get("open_gravity_goal_sha256")
        == "f083db7acb27896b7ede7dec3e415c7ebf5e3211dd1781699985c120e2db3106",
        "open-gravity goal binding changed",
    )
    _require(
        goal.get("baseline_roadmap_sha256")
        == "94305ae9e037200dabc098781a31ee060f0296719621ee262440efb5060f9d79",
        "baseline roadmap binding changed",
    )
    candidate = config.get("candidate_contract", {})
    _require(
        candidate
        == {
            "twell_live_cards": 400,
            "gp01_live_cards": 7,
            "total_live_cards": 407,
            "exact_parameter_cells": 2486,
            "source_ready_parameter_cells": {"GALAXIES": 179, "CLUSTERS": 1669},
            "source_ready_concepts": {"GALAXIES": 61, "CLUSTERS": 128},
            "all_five_lanes_required": True,
            "post_freeze_formula_changes": 0,
            "revision_budget": 0,
        },
        "candidate contract changed",
    )
    data = config.get("data_contract", {})
    _require(
        data.get("sparc_objects") == 139 and data.get("sparc_rows") == 2720, "SPARC scope changed"
    )
    _require(data.get("xcop_response_rows") == 184, "X-COP response-row scope changed")
    _require(
        data.get("sparc_pilot_count") == 28 and data.get("sparc_full_count") == 111,
        "SPARC split changed",
    )
    _require(
        data.get("xcop_objects")
        == ["A1644", "A1795", "A2142", "A2255", "A2319", "A3266", "A85", "ZW1215"],
        "X-COP allowlist changed",
    )
    _require(data.get("xcop_pilot") == ["A85", "A3266"], "X-COP pilot changed")
    _require(len(config.get("nuisance_cases", [])) == 6, "nuisance grid changed")
    xcop_nuisance = [row for row in config["nuisance_cases"] if row.get("domain") == "CLUSTERS"]
    _require(
        [row.get("missing_stellar_to_gas_mass_ratio") for row in xcop_nuisance] == [0.1, 0.1, 0.1],
        "shared missing-stellar rule changed",
    )
    _require(len(config.get("transformations", [])) == 2, "control transforms changed")
    _require(len(config.get("comparators", [])) == 13, "comparator inventory changed")
    metrics = config.get("metrics", {})
    _require(metrics.get("minimum_meaningful_improvement") == 0.02, "improvement gate changed")
    _require(metrics.get("minimum_galaxy_support") == 84, "galaxy breadth changed")
    _require(metrics.get("minimum_cluster_support") == 6, "cluster breadth changed")
    _require(
        config.get("adjudication_rule")
        == {
            "nuisance_aggregation": "WORST_CASE_OF_THREE_PER_DOMAIN",
            "candidate_must_beat_each_domain_strongest_executable_comparator_in_every_nuisance_case": True,
            "identity_transformation_is_scientific_score": True,
            "radial_factor_reversal_is_negative_control_only": True,
            "same_exact_parameter_cell_required_across_galaxies_and_clusters": True,
            "zero_survivors_allowed": True,
        },
        "adjudication rule changed",
    )
    multiplicity = config.get("multiplicity", {})
    _require(multiplicity.get("legacy_ledger_incomplete") is True, "legacy caveat removed")
    _require(
        multiplicity.get("nominal_global_discovery_p_value_allowed") is False,
        "global p-value overclaim enabled",
    )
    _require(
        multiplicity.get("maximum_promotion_label") == "DEVELOPMENT_SIGNAL", "claim cap changed"
    )
    _require(multiplicity.get("confirmation_K") == 1, "confirmation K changed")
    ceiling = config.get("execution_ceiling", {})
    _require(
        ceiling.get("xcop_response_rows_scored") == 184,
        "X-COP scored-row ceiling changed",
    )
    _require(
        {
            key: ceiling.get(key)
            for key in (
                "scientific_response_files",
                "scientific_source_files",
                "unique_local_payload_files",
                "committed_sparc_blob_verification_reads_per_pass",
                "local_payload_read_operations_per_execute",
                "local_payload_read_operations_per_check_result",
                "local_payload_read_operations_after_execute_and_check_result",
            )
        }
        == {
            "scientific_response_files": 17,
            "scientific_source_files": 13,
            "unique_local_payload_files": 30,
            "committed_sparc_blob_verification_reads_per_pass": 1,
            "local_payload_read_operations_per_execute": 31,
            "local_payload_read_operations_per_check_result": 31,
            "local_payload_read_operations_after_execute_and_check_result": 62,
        },
        "local payload read accounting changed",
    )
    for field in ("network_calls", "model_calls", "paid_calls"):
        _require(ceiling.get(field) == 0, f"forbidden execution ceiling changed: {field}")
    _require(
        config.get("scientific_input_bindings")
        == {
            "sparc": {
                "dataset_path": "configs/sparc_rotation_curves_full_v1.json",
                "dataset_raw_sha256": (
                    "dde80c7fc72974358b1370e1978726b87fe1a4048f0880ae79cf513e260a7cf1"
                ),
                "dataset_commit": "92bc8bfcdc31714d9b9f69b86b44dc3920613350",
                "objects_in_container": 175,
                "rows_in_container": 3391,
                "objects_scored": 139,
                "rows_scored": 2720,
            },
            "xcop": {
                "raw_root": (
                    "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw"
                ),
                "input_contract_sha256": (
                    "b77c4bcbdfc57f33499e4002f84dd3eb162a38517242373b95d06b912706e0b5"
                ),
                "unique_files": 29,
                "total_bytes": 538560,
                "required_roles": ["density", "pressure", "stellar_mass", "temperature"],
            },
        },
        "scientific input bindings changed",
    )
    _require(
        config.get("preparation_access_disclosure")
        == {
            "authorized_opaque_file_hash_reads": 1,
            "path": "configs/sparc_rotation_curves_full_v1.json",
            "reason": (
                "The primary agent computed the already-known raw SHA-256 while sealing "
                "the post-run verifier before access intent."
            ),
            "response_rows_decoded_or_inspected": 0,
            "scientific_values_logged": 0,
            "scores_computed": 0,
            "candidate_selection_events": 0,
            "affects_target_blindness": False,
            "part_of_response_scored_campaign": False,
        },
        "preparation access disclosure changed",
    )
    _require(
        config.get("artifact_contract")
        == {
            "mandatory_campaign_artifacts": list(MANDATORY_CAMPAIGN_ARTIFACTS),
            "galaxy_dashboards": 139,
            "cluster_dashboards": 8,
            "total_dashboards": 147,
            "total_artifacts": 156,
            "exact_unique_path_set_required": True,
            "canonical_json_required": True,
            "schema_and_manifest_crosslinks_required": True,
            "check_result_recomputes_all_scores_and_artifacts_from_frozen_local_inputs": True,
            "diagnostic_recompute_is_not_a_second_campaign_or_selection": True,
            "supported_validation_entrypoint": "FRESH_PYTHON_PROCESS_OFFICIAL_CLI_ONLY",
            "in_process_monkeypatch_resistance_claim": False,
        },
        "result artifact contract changed",
    )
    claims = config.get("claim_ceiling", {})
    _require(claims.get("maximum_label") == "DEVELOPMENT_SIGNAL", "claim ceiling changed")
    _require(
        all(value is False for key, value in claims.items() if key.endswith("_claim")),
        "scientific claim ceiling was promoted",
    )
    _exact_paths(config)


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    expected = (_repo_root() / CONFIG_PATH).resolve()
    target = (repo / CONFIG_PATH).resolve()
    _require(target == expected, "only the canonical campaign config path is allowed")
    _validate_local_artifact_integrity()
    config = _read_json(target)
    validate_config(config)
    return config


def _git_show(repo: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _binding_paths(binding: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in binding.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        hash_key = key.removesuffix("_path") + "_sha256"
        if hash_key in binding and isinstance(binding[hash_key], str):
            rows.append((value, str(binding[hash_key])))
    return rows


def verify_dependency_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    repo = root.resolve()
    verified: list[dict[str, Any]] = []
    for role, binding in config["dependency_bindings"].items():
        commit = str(binding["commit"])
        _require(
            len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
            f"unsealed commit: {role}",
        )
        for relative, expected_sha in _binding_paths(binding):
            _require(len(expected_sha) == 64, f"unsealed dependency hash: {role}:{relative}")
            path = repo / relative
            actual = file_sha256(path)
            _require(actual == expected_sha, f"dependency bytes changed: {role}:{relative}")
            committed = hashlib.sha256(_git_show(repo, commit, relative)).hexdigest()
            _require(committed == expected_sha, f"commit binding changed: {role}:{relative}")
            verified.append({"role": role, "path": relative, "sha256": actual, "commit": commit})
    for goal_key, sha_key in (
        ("open_gravity_goal_path", "open_gravity_goal_sha256"),
        ("baseline_roadmap_path", "baseline_roadmap_sha256"),
    ):
        path = Path(config["goal_bindings"][goal_key])
        _require(
            file_sha256(path) == config["goal_bindings"][sha_key],
            f"goal binding changed: {goal_key}",
        )
    return {"verified_files": len(verified), "bindings": verified}


def _load_live_cards(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    twell_rows, _stream, twell_receipt = twell.build_packet(root)
    adapter_config = adapter.load_config(root)
    adapter_catalog = adapter.typed_mechanism_card_catalog(root, adapter_config)
    gp01_wrappers = [
        row
        for row in adapter_catalog["cards"]
        if str(row["card"]["stable_concept_id"]).startswith("GP01-")
    ]
    _require(len(twell_rows) == 400, "TWELL live-card count changed")
    _require(len(gp01_wrappers) == 7, "GP01 live-card count changed")
    cards = [dict(row["card"]) for row in twell_rows] + [dict(row["card"]) for row in gp01_wrappers]
    _require(
        len(cards) == len({str(card["card_id"]) for card in cards}) == 407, "live cards collide"
    )
    return cards, {
        "twell_rows": twell_rows,
        "twell_receipt": twell_receipt,
        "gp01_wrappers": gp01_wrappers,
        "adapter_catalog": adapter_catalog,
    }


def _load_source_projection(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    _require(root.resolve() == _repo_root(), "only canonical source matrix root allowed")
    source_config = sources.load_config()
    predecessor = sources.load_predecessor(source_config)
    sources.require_final_gates(source_config)
    concept_rows = list(sources.iter_manifest_concept_domain_rows(source_config, predecessor))
    cell_rows = list(sources.iter_manifest_parameter_cell_domain_rows(source_config, predecessor))
    _require(len(concept_rows) == 1680, "concept-domain source projection changed")
    _require(len(cell_rows) == 9944, "parameter-domain source projection changed")
    concept_map = {
        (str(row["mechanism_id"]), str(row["registry_domain"])): row for row in concept_rows
    }
    cell_map = {
        (
            str(row["mechanism_id"]),
            str(row["parameter_cell_id"]),
            str(row["registry_domain"]),
        ): row
        for row in cell_rows
    }
    return source_config, predecessor, concept_map, cell_map


def _gp01_wrapper_map(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["card"]["stable_concept_id"]): row for row in packet["gp01_wrappers"]}


def _source_ready(row: Mapping[str, Any]) -> bool:
    return str(row["concept_readiness"]).startswith("SOURCE_READY_")


def _twell_domain_execution(
    concept_id: str,
    concept_map: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        row = concept_map[(concept_id, domain)]
        source_hash = str(row["source_contract_sha256"])
        readiness = str(row["concept_readiness"])
        if _source_ready(row):
            result[domain] = {
                "eligible": True,
                "execution_disposition": "SEALED_UNOPENED_FOR_SCORING",
                "scored": False,
                "source_contract_sha256": source_hash,
            }
        elif readiness == "KNOWN_REWRITE_NONINDEPENDENT":
            result[domain] = {
                "eligible": False,
                "execution_disposition": "KNOWN_REWRITE_NONINDEPENDENT",
                "scored": False,
                "source_contract_sha256": source_hash,
            }
        elif readiness == "QUARANTINED":
            result[domain] = {
                "eligible": False,
                "execution_disposition": "QUARANTINED",
                "scored": False,
                "source_contract_sha256": source_hash,
            }
        elif readiness == "THEORY_ONLY":
            result[domain] = {
                "eligible": False,
                "execution_disposition": "THEORY_ONLY",
                "scored": False,
                "source_contract_sha256": source_hash,
            }
        else:
            result[domain] = {
                "eligible": False,
                "execution_disposition": "SOURCE_BLOCKED",
                "scored": False,
                "source_contract_sha256": source_hash,
            }
    return result


def _gp01_domain_execution(
    concept_id: str,
    wrapper: Mapping[str, Any],
    concept_map: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(dict(wrapper["domain_execution"]))
    if concept_id == "GP01-L":
        for domain in ("GALAXIES", "CLUSTERS"):
            result[domain] = {
                "eligible": True,
                "execution_disposition": "SEALED_UNOPENED_FOR_SCORING",
                "scored": False,
                "source_contract_sha256": concept_map[(concept_id, domain)][
                    "source_contract_sha256"
                ],
            }
    elif concept_id == "GP01-ELLIPTIC":
        result["CLUSTERS"] = {
            "eligible": True,
            "execution_disposition": "SEALED_UNOPENED_FOR_SCORING",
            "scored": False,
            "source_contract_sha256": concept_map[(concept_id, "CLUSTERS")][
                "source_contract_sha256"
            ],
        }
    return result


def _candidate_status(domain_execution: Mapping[str, Mapping[str, Any]]) -> str:
    dispositions = {str(row["execution_disposition"]) for row in domain_execution.values()}
    if "SEALED_UNOPENED_FOR_SCORING" in dispositions:
        return "READY_FOR_RESPONSE_SCORING"
    if "KNOWN_REWRITE_NONINDEPENDENT" in dispositions:
        return "KNOWN_REWRITE_NONINDEPENDENT"
    if "QUARANTINED" in dispositions:
        return "QUARANTINED_REVISION_REQUIRED"
    if "SOURCE_BLOCKED" in dispositions:
        return "SOURCE_BLOCKED"
    return "REGISTERED_THEORY_ONLY"


def _candidate_versions(
    packet: Mapping[str, Any],
    concept_map: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in packet["twell_rows"]:
        item = copy.deepcopy(dict(row["manifest_input"]))
        concept_id = str(item["candidate_id"])
        item["domain_execution"] = _twell_domain_execution(concept_id, concept_map)
        item["candidate_status"] = _candidate_status(item["domain_execution"])
        if (
            item["candidate_status"] == "SOURCE_BLOCKED"
            and row["schema_admission"]["status"] == "READY_FOR_THEORY_GATES"
        ):
            for domain in DOMAINS:
                item["domain_execution"][domain] = {
                    **item["domain_execution"][domain],
                    "execution_disposition": "THEORY_ONLY",
                }
            item["candidate_status"] = "REGISTERED_THEORY_ONLY"
        rows.append(item)
    for concept_id, wrapper in sorted(_gp01_wrapper_map(packet).items()):
        card = wrapper["card"]
        domain_execution = _gp01_domain_execution(concept_id, wrapper, concept_map)
        rows.append(
            {
                "candidate_id": concept_id,
                "card_id": card["card_id"],
                "semantic_version": card["semantic_version"],
                "anonymous_formula_id": "PENDING",
                "lane": wrapper["lane_hint"],
                "candidate_status": _candidate_status(domain_execution),
                "scientific_status": card["scientific_status"],
                "identity_class": card["identity_class"],
                "mechanism_kind": card["action_or_equations"]["kind"],
                "mechanism_executable": card["action_or_equations"]["executable"],
                "card_sha256": wrapper["card_sha256"],
                "formula_sha256": wrapper["formula_sha256"],
                "configuration_sha256": wrapper["configuration_sha256"],
                "equivalence_family_id": wrapper["formula_equivalence_family_id"],
                "equivalence_fingerprint_sha256": wrapper["equivalence_fingerprint_sha256"],
                "domain_execution": domain_execution,
            }
        )
    rows.sort(key=lambda item: str(item["candidate_id"]))
    for index, row in enumerate(rows, start=1):
        row["anonymous_formula_id"] = f"F{index:04d}"
    _require(len(rows) == 407, "candidate row count changed")
    return rows


def _gp01_elliptic_cell_id(parameters: Mapping[str, Any]) -> str:
    return (
        f"GP01E-n{parameters['n']}-A{parameters['A_max']:g}"
        f"-rho{parameters['rho_ratio']:g}-T{parameters['tide_ratio']:g}"
        f"-q{parameters['q']}-p{parameters['tide_power']}-L{parameters['L_ratio']:g}"
    )


def _cell_source_contracts(
    concept_id: str,
    cell_id: str,
    cell_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        domain: {
            "readiness": cell_map[(concept_id, cell_id, domain)]["parameter_cell_readiness"],
            "source_contract_sha256": cell_map[(concept_id, cell_id, domain)][
                "source_contract_sha256"
            ],
        }
        for domain in DOMAINS
    }


def _parameter_cells(
    packet: Mapping[str, Any],
    cell_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in packet["twell_rows"]:
        concept_id = str(concept["concept_id"])
        for cell in concept["card"]["parameter_cells"]:
            cell_id = str(cell["cell_id"])
            rows.append(
                {
                    "cell_id": cell_id,
                    "exact_value_or_rule": {
                        "concept_id": concept_id,
                        "parameter": cell["parameter"],
                        "value": cell["value"],
                        "unit": cell["unit"],
                        "domain_sources": _cell_source_contracts(concept_id, cell_id, cell_map),
                    },
                    "frozen": True,
                }
            )
    for concept_id, prefix in (("GP01-L", "GP01L"), ("GP01-AQUAL", "GP01AQUAL")):
        for n in (1, 2, 4):
            cell_id = f"{prefix}-n{n}"
            rows.append(
                {
                    "cell_id": cell_id,
                    "exact_value_or_rule": {
                        "concept_id": concept_id,
                        "value": {"n": n, "a_star_m_s2": 1.2e-10},
                        "unit": "n:1,a_star:m/s^2",
                        "domain_sources": _cell_source_contracts(concept_id, cell_id, cell_map),
                    },
                    "frozen": True,
                }
            )
    for parameters in adapter._gp01_cell_parameters():
        cell_id = _gp01_elliptic_cell_id(parameters)
        rows.append(
            {
                "cell_id": cell_id,
                "exact_value_or_rule": {
                    "concept_id": "GP01-ELLIPTIC",
                    "value": parameters,
                    "unit": "n,A_max,rho_ratio,tide_ratio,q,tide_power,L_ratio:dimensionless",
                    "domain_sources": _cell_source_contracts("GP01-ELLIPTIC", cell_id, cell_map),
                },
                "frozen": True,
            }
        )
    rows.sort(key=lambda row: str(row["cell_id"]))
    _require(
        len(rows) == len({str(row["cell_id"]) for row in rows}) == 2486,
        "parameter-cell count changed",
    )
    return rows


def _object_mapping(predecessor: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    salt = str(predecessor["partition_design"]["salt_label"])
    entries = [("GALAXIES", str(name)) for name in predecessor["objects"]["SPARC"]] + [
        ("CLUSTERS", str(name)) for name in predecessor["objects"]["XCOP"]
    ]
    entries.sort(
        key=lambda row: (hashlib.sha256(f"{salt}|{row[0]}|{row[1]}".encode()).hexdigest(), row)
    )
    rows = [
        {"domain": domain, "object": name, "anonymous_object_id": f"O{index:04d}"}
        for index, (domain, name) in enumerate(entries, start=1)
    ]
    mapping = {f"{row['domain']}:{row['object']}": row["anonymous_object_id"] for row in rows}
    _require(len(rows) == len(mapping) == 147, "object mapping changed")
    return rows, mapping


def _named_item(item_id: str, definition: str) -> dict[str, str]:
    return {
        "item_id": item_id,
        "definition": definition,
        "implementation_sha256": content_sha256({"item_id": item_id, "definition": definition}),
    }


def _partition(
    partition_id: str,
    role: str,
    ids: Sequence[str],
    data_contract: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(map(str, ids))
    return {
        "partition_id": partition_id,
        "role": role,
        "anonymous_object_ids": ordered,
        "object_ledger_sha256": registry.partition_object_ledger_sha256(ordered),
        "data_contract_sha256": content_sha256(data_contract),
    }


def _multiplicity_counts(
    candidates: Sequence[Mapping[str, Any]],
    parameter_cells: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    object_subsets: int,
    observables: int,
    metrics: int,
    comparators: int,
    stopping_decisions: int,
    selection_stages: int,
) -> dict[str, int]:
    del comparators
    planned = [
        row
        for row in candidates
        if any(
            domain["execution_disposition"] == "SEALED_UNOPENED_FOR_SCORING"
            for domain in row["domain_execution"].values()
        )
    ]
    planned_domains = sum(
        domain["execution_disposition"] == "SEALED_UNOPENED_FOR_SCORING"
        for row in candidates
        for domain in row["domain_execution"].values()
    )
    return {
        "response_scored_campaigns": 0,
        "response_planned_campaigns": 1,
        "adaptive_generations": 0,
        "concepts": len({str(row["candidate_id"]) for row in candidates}),
        "registered_candidate_rows": len(candidates),
        "equivalence_families": len({str(row["equivalence_family_id"]) for row in candidates}),
        "formula_variants": len({str(row["formula_sha256"]) for row in candidates}),
        "parameter_cells": len(parameter_cells),
        "hyperparameter_cells": 0,
        "nuisance_scenarios": len(config["nuisance_cases"]),
        "transformations": len(config["transformations"]),
        "object_subsets": object_subsets,
        "observables": observables,
        "metrics": metrics,
        "repairs": 0,
        "stopping_decisions": stopping_decisions,
        "residual_inspired_branches": 0,
        "selection_stages": selection_stages,
        "response_planned_formula_variants": len({str(row["formula_sha256"]) for row in planned}),
        "response_planned_domain_executions": int(planned_domains),
        "response_scored_formula_variants": 0,
        "response_scored_domain_executions": 0,
    }


def build_manifest(root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = _repo_root() if root is None else root.resolve()
    config = load_config(repo)
    verify_dependency_bindings(repo, config)
    cards, packet = _load_live_cards(repo)
    _source_config, predecessor, concept_map, cell_map = _load_source_projection(repo)
    candidates = _candidate_versions(packet, concept_map)
    parameter_cells = _parameter_cells(packet, cell_map)
    object_rows, object_map = _object_mapping(predecessor)

    sparc_names = list(map(str, predecessor["objects"]["SPARC"]))
    sparc_pilot = list(map(str, predecessor["partition_design"]["SPARC_pilot"]))
    sparc_full = sorted(set(sparc_names) - set(sparc_pilot))
    xcop_pilot = list(map(str, predecessor["partition_design"]["XCOP_pilot"]))
    xcop_full = list(map(str, predecessor["partition_design"]["XCOP_validation"]))
    pilot_ids = [object_map[f"GALAXIES:{name}"] for name in sparc_pilot] + [
        object_map[f"CLUSTERS:{name}"] for name in xcop_pilot
    ]
    full_ids = [object_map[f"GALAXIES:{name}"] for name in sparc_full] + [
        object_map[f"CLUSTERS:{name}"] for name in xcop_full
    ]
    all_ids = [row["anonymous_object_id"] for row in object_rows]

    source_contract = {
        "source_matrix_receipt_content_sha256": config["dependency_bindings"]["source_matrix"][
            "receipt_content_sha256"
        ],
        "objects": predecessor["object_ledger_seals"],
        "response_fields_forbidden_during_source_compile": True,
    }
    response_contract = {
        "development_only": True,
        "sparc_rows": 2720,
        "xcop_clusters": config["data_contract"]["xcop_objects"],
        "group_lensing_confirmation_independent_forbidden": True,
    }
    source_partitions = [_partition("SOURCE-ALL-147", "SOURCE_ONLY", all_ids, source_contract)]
    response_partitions = [
        _partition("DEVELOPMENT-PILOT-30", "DEVELOPMENT_PILOT", pilot_ids, response_contract),
        _partition("DEVELOPMENT-FULL-117", "DEVELOPMENT_FULL", full_ids, response_contract),
        _partition("CONFIRMATION-K1-SEALED", "CONFIRMATION_SEALED", ["O9001"], {"sealed": True}),
        _partition("INDEPENDENT-SEALED", "INDEPENDENT_SEALED", ["O9002"], {"sealed": True}),
    ]

    transformations = [
        _named_item(str(row["id"]), str(row["definition"])) for row in config["transformations"]
    ]
    object_subsets = [
        _named_item("SPARC-PILOT-28", "Frozen 28-object target-blind SPARC pilot."),
        _named_item("SPARC-FULL-111", "Disjoint 111-object SPARC full-development partition."),
        _named_item("XCOP-PILOT-2", "Frozen A85/A3266 pilot."),
        _named_item("XCOP-FULL-6", "Disjoint six-cluster full-development partition."),
        _named_item("ALL-GALAXIES-139", "Every admitted development galaxy."),
        _named_item("ALL-CLUSTERS-8", "Every authorized development cluster."),
    ]
    observables = [
        _named_item("SPARC-V", "Published circular-speed response with published uncertainty."),
        _named_item("XCOP-P", "Published pressure response with shared X-ray ancestry disclosed."),
        _named_item(
            "XCOP-T", "Published temperature response with shared X-ray ancestry disclosed."
        ),
    ]
    metric_items = [
        _named_item("OBJECT-MSSR", config["metrics"]["primary"]),
        _named_item("EQUAL-CLUSTER-OBS", config["metrics"]["cluster_aggregation"]),
        _named_item("WORST-OBJECT", "Maximum per-object mean squared standardized residual."),
        _named_item(
            "LOO-STABILITY",
            "Minimum leave-one-object-out improvement over the strongest comparator.",
        ),
    ]
    comparator_items = [
        _named_item(
            str(row["id"]),
            f"{row['status']}; exact implementation binding is in the comparator ledger.",
        )
        for row in config["comparators"]
    ]
    stopping = [
        _named_item(
            "ONE-CAMPAIGN",
            "The session ends after this campaign adjudicates, including zero survivors.",
        ),
        _named_item(
            "NO-POST-PILOT-REPAIR",
            "No equation, parameter grid, or closure changes after pilot scores exist.",
        ),
        _named_item(
            "BLOCKED-IS-NOT-FAILURE", "Missing independent cause-side inputs remain source-blocked."
        ),
    ]
    stages = [
        _named_item(
            "PILOT",
            "Score all eligible frozen cells on the disjoint pilot objects without revealing ranks.",
        ),
        _named_item(
            "FULL-DEVELOPMENT", "Score unchanged cells on the disjoint full-development objects."
        ),
        _named_item(
            "CROSS-DOMAIN-ADJUDICATION",
            "Apply shared-cell, breadth, nuisance, LOO, and claim-ceiling gates.",
        ),
    ]

    registry_config = registry.load_config()
    manifest_schema = registry.load_schemas(repo)["campaign_manifest"]
    blind = registry_config["target_blind_contract"]
    current = _multiplicity_counts(
        candidates,
        parameter_cells,
        config,
        object_subsets=len(object_subsets),
        observables=len(observables),
        metrics=len(metric_items),
        comparators=len(comparator_items),
        stopping_decisions=len(stopping),
        selection_stages=len(stages),
    )
    zero = {key: 0 for key in current}
    formula_mapping = [
        {"candidate_id": row["candidate_id"], "anonymous_formula_id": row["anonymous_formula_id"]}
        for row in candidates
    ]
    lane_counts = Counter(str(row["lane"]) for row in candidates)
    manifest: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA,
        "manifest_id": "OPEN-GRAVITY-CAMPAIGN-MANIFEST-v1",
        "campaign_id": config["campaign_id"],
        "semantic_version": config["semantic_version"],
        "manifest_state": "FROZEN_UNRUN",
        "frozen_at_utc": config["frozen_at_utc"],
        "frozen_before_response_access": True,
        "response_scored_campaign": True,
        "registry_binding": {
            "registry_id": registry.REGISTRY_ID,
            "semantic_version": registry.REGISTRY_VERSION,
            "foundation_receipt_sha256": registry.build_receipt(repo)["content_sha256"],
            "mechanism_card_set_sha256": registry.mechanism_card_set_sha256(cards),
            "equivalence_ledger_sha256": registry.campaign_equivalence_ledger_sha256(candidates),
            "trusted_session_contract_sha256": registry.trusted_session_contract_sha256(
                registry_config
            ),
            "twell_400_ids_sha256": registry.TWELL_IDS_SHA256,
        },
        "candidate_versions": candidates,
        "data_roles_and_splits": {
            "source_partitions": source_partitions,
            "response_partitions": response_partitions,
            "pilot_full_relation": "DISJOINT",
            "confirmation_forbidden_in_campaign": True,
        },
        "parameter_cells": parameter_cells,
        "hyperparameter_cells": [],
        "nuisance_cases": [
            {"cell_id": row["cell_id"], "exact_value_or_rule": row, "frozen": True}
            for row in config["nuisance_cases"]
        ],
        "adaptive_generation_ids": [],
        "transformations": transformations,
        "object_subsets": object_subsets,
        "observables": observables,
        "metrics": metric_items,
        "comparators": comparator_items,
        "repairs": [],
        "stopping_decisions": stopping,
        "residual_inspired_branch_ids": [],
        "selection_stages": stages,
        "correction_method": {
            "method_id": "LEGACY-LOWER-BOUND-DEVELOPMENT-CAP-v1",
            "exact_rule": (
                "Charge at least 450267929 historical search units; because the legacy ledger is incomplete, "
                "report raw development losses and breadth only, no nominal global discovery p-value, and cap "
                "promotion at DEVELOPMENT_SIGNAL."
            ),
            "selection_adjusted_reporting": True,
            "global_sequential_evidence_budget_units": int(
                config["multiplicity"]["known_legacy_lower_bound"]
            )
            + 1,
        },
        "global_multiplicity_ledger": {
            "ledger_id": registry.MULTIPLICITY_LEDGER_ID,
            "campaign_sequence": 1,
            "previous_manifest_sha256": "GENESIS",
            "never_resets": True,
            "counts_before": zero,
            "counts_this_campaign": current,
            "counts_after": current,
        },
        "promotion_thresholds": {
            "shared_constants_required": True,
            "minimum_meaningful_improvement": config["metrics"]["minimum_meaningful_improvement"],
            "selection_adjusted_evidence_threshold": 0.05,
            "leave_one_object_out_minimum": config["metrics"][
                "leave_one_object_out_minimum_improvement"
            ],
            "minimum_object_breadth": config["metrics"]["minimum_galaxy_support"],
            "minimum_domain_breadth": config["metrics"]["minimum_domain_breadth"],
        },
        "worst_case_and_subgroup_ceilings": [
            {
                "ceiling_id": "WORST-OBJECT-RATIO",
                "metric_id": "WORST-OBJECT",
                "scope": "every scored object",
                "maximum": config["metrics"]["maximum_worst_object_loss_ratio"],
            },
            {
                "ceiling_id": "SUBGROUP-RATIO",
                "metric_id": "OBJECT-MSSR",
                "scope": "pilot/full and every declared domain subgroup",
                "maximum": config["metrics"]["maximum_subgroup_loss_ratio"],
            },
        ],
        "budgets": {
            "lane_candidate_limits": {
                "CORE": lane_counts["CORE"],
                "ADJACENT": lane_counts["ADJACENT"],
                "ORTHOGONAL": lane_counts["ORTHOGONAL"],
                "RIVALS_CONTROLS": lane_counts["RIVALS_CONTROLS"],
                "WILDCARD": lane_counts["WILDCARD"],
            },
            "revision_limit": 0,
            "compute_cost_ceiling": "Local CPU only; exact finite cells in execution_ceiling; no adaptive generation.",
            "network_cost_ceiling": 0,
            "model_cost_ceiling": 0,
            "paid_cost_ceiling": 0,
        },
        "confirmation": {
            "K": 1,
            "evidence_budget_units_total": 1,
            "slots": [
                {
                    "slot_id": "K1",
                    "evidence_budget_units": 1,
                    "sealed": True,
                    "opened": False,
                    "candidate_id": None,
                }
            ],
            "opened_slots": 0,
            "independent_reduction_required": True,
        },
        "target_blind_contract": {
            "anonymous_formula_identifiers": blind["anonymous_formula_identifiers"],
            "anonymous_object_identifiers": blind["anonymous_object_identifiers"],
            "target_class_switches_allowed": blind["target_class_switches_allowed"],
            "object_specific_gravity_tuning_allowed": blind[
                "object_specific_gravity_tuning_allowed"
            ],
            "score_feedback_to_formula_authors_before_batch_close": blind[
                "score_feedback_to_formula_authors_before_batch_close"
            ],
            "theory_author_may_adjudicate_own_candidate": blind[
                "theory_author_may_adjudicate_own_candidate"
            ],
            "legitimate_source_metadata_visible": blind["legitimate_source_metadata_visible"],
            "forbidden_generation_inputs": blind["forbidden_generation_inputs"],
            "sealed_formula_mapping_sha256": content_sha256(formula_mapping),
            "sealed_object_mapping_sha256": content_sha256(object_rows),
        },
        "session_terminal_contract": {
            "session_id": registry.TRUSTED_SESSION_ID,
            "trusted_session_contract_sha256": registry.trusted_session_contract_sha256(
                registry_config
            ),
            "terminal_ledger_contract_sha256": registry.terminal_ledger_contract_sha256(
                registry_config
            ),
            "campaign_execution_authority": "WITHHELD_PENDING_PERSISTED_TERMINAL_LEDGER",
            "response_scored_campaign_limit": 1,
            "response_scored_campaign_ordinal": 1,
            "automatic_second_campaign_allowed": False,
            "on_adjudication": "SESSION_TERMINAL",
            "post_freeze_new_or_repaired_idea_destination": registry.IDEA_RESERVOIR_ID,
            "zero_survivors_allowed": True,
        },
        "zero_access_at_freeze": copy.deepcopy(ZERO_ACCESS),
        "manifest_content_sha256": "",
    }
    manifest["manifest_content_sha256"] = registry.campaign_manifest_sha256(manifest)
    registry.validate_campaign_manifest(
        manifest,
        manifest_schema,
        registry_config,
        mechanism_cards=cards,
        root=repo,
    )
    context = {
        "cards": cards,
        "packet": packet,
        "source_predecessor": predecessor,
        "concept_map": concept_map,
        "cell_map": cell_map,
        "object_rows": object_rows,
        "object_map": object_map,
    }
    return manifest, context


def build_terminal_ledger(manifest: Mapping[str, Any]) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema_version": TERMINAL_SCHEMA,
        "session_id": registry.TRUSTED_SESSION_ID,
        "campaign_id": manifest["campaign_id"],
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "campaign_ordinal": 1,
        "previous_entry_sha256": "GENESIS",
        "adjudication_state": "RESERVED_UNRUN",
        "session_terminal": True,
        "execution_authority": "GRANTED_EXACTLY_ONCE_FOR_RESERVED_CAMPAIGN",
        "response_scope": "LOCAL_DEVELOPMENT_ONLY_NO_GROUP_LENSING_CONFIRMATION_INDEPENDENT",
        "automatic_second_campaign_allowed": False,
        "zero_survivors_allowed": True,
        "created_at_utc": "2026-08-30T19:00:00Z",
        "ledger_content_sha256": "",
    }
    ledger["ledger_content_sha256"] = _self_hash(ledger, "ledger_content_sha256")
    return ledger


def validate_terminal_ledger(ledger: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected = build_terminal_ledger(manifest)
    _require(dict(ledger) == expected, "terminal ledger changed")
    _require(ledger["session_terminal"] is True, "session is not terminally reserved")
    _require(
        ledger["execution_authority"] == "GRANTED_EXACTLY_ONCE_FOR_RESERVED_CAMPAIGN",
        "terminal ledger does not grant the exact reserved execution",
    )


def build_preflight(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    manifest, _context = build_manifest(repo)
    ledger = build_terminal_ledger(manifest)
    _require((repo / TEST_PATH).exists(), "campaign tests do not exist")
    preflight: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "decision": "READY_FOR_EXACTLY_ONE_LOCAL_DEVELOPMENT_EXECUTION_AFTER_COMMIT",
        "status": "FROZEN_UNRUN_ZERO_RESPONSE_ROW_ACCESS_ONE_DISCLOSED_OPAQUE_HASH_READ",
        "package": {
            "config": {"path": CONFIG_PATH.as_posix(), "sha256": file_sha256(repo / CONFIG_PATH)},
            "module": {"path": MODULE_PATH.as_posix(), "sha256": file_sha256(repo / MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": file_sha256(repo / TEST_PATH)},
        },
        "manifest": {
            "path": MANIFEST_PATH.as_posix(),
            "sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
            "content_sha256": manifest["manifest_content_sha256"],
            "candidate_count": len(manifest["candidate_versions"]),
            "parameter_cell_count": len(manifest["parameter_cells"]),
            "planned_formula_variants": manifest["global_multiplicity_ledger"][
                "counts_this_campaign"
            ]["response_planned_formula_variants"],
            "planned_domain_executions": manifest["global_multiplicity_ledger"][
                "counts_this_campaign"
            ]["response_planned_domain_executions"],
        },
        "terminal_ledger": {
            "path": TERMINAL_LEDGER_PATH.as_posix(),
            "sha256": hashlib.sha256(canonical_bytes(ledger)).hexdigest(),
            "content_sha256": ledger["ledger_content_sha256"],
            "session_terminal": True,
            "campaign_ordinal": 1,
        },
        "source_ready": {
            "GALAXIES_parameter_cells": 179,
            "CLUSTERS_parameter_cells": 1669,
            "GROUPS": "SOURCE_BLOCKED",
            "LENSING": "SOURCE_BLOCKED",
        },
        "access": copy.deepcopy(ZERO_ACCESS),
        "preparation_access_disclosure": copy.deepcopy(
            load_config(repo)["preparation_access_disclosure"]
        ),
        "cost": {"network": 0, "model": 0, "paid": 0, "external_usd": 0.0},
        "claim_ceiling": copy.deepcopy(load_config(repo)["claim_ceiling"]),
        "preflight_content_sha256": "",
    }
    preflight["preflight_content_sha256"] = _self_hash(preflight, "preflight_content_sha256")
    return preflight


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return "EXISTING_IDENTICAL" if path.read_bytes() == payload else "EXISTING_DIFFERENT"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return "EXISTING_IDENTICAL" if path.read_bytes() == payload else "EXISTING_DIFFERENT"
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except PermissionError:
            _require(os.name == "nt", "directory durability flush failed")
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_preflight(root: Path | None = None) -> dict[str, str]:
    repo = _repo_root() if root is None else root.resolve()
    manifest, _context = build_manifest(repo)
    ledger = build_terminal_ledger(manifest)
    preflight = build_preflight(repo)
    statuses = {
        "manifest": _atomic_no_clobber(repo / MANIFEST_PATH, canonical_bytes(manifest)),
        "terminal_ledger": _atomic_no_clobber(repo / TERMINAL_LEDGER_PATH, canonical_bytes(ledger)),
        "preflight": _atomic_no_clobber(repo / PREFLIGHT_PATH, canonical_bytes(preflight)),
    }
    _require(
        all(status != "EXISTING_DIFFERENT" for status in statuses.values()),
        "preflight no-clobber refusal",
    )
    return statuses


def check_preflight(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    _require(repo == _repo_root(), "only the canonical repository root is allowed")
    expected_manifest, _context = build_manifest(repo)
    expected_ledger = build_terminal_ledger(expected_manifest)
    expected_preflight = build_preflight(repo)
    for relative, expected in (
        (MANIFEST_PATH, expected_manifest),
        (TERMINAL_LEDGER_PATH, expected_ledger),
        (PREFLIGHT_PATH, expected_preflight),
    ):
        target = (repo / relative).resolve()
        _require(target == (_repo_root() / relative).resolve(), "noncanonical check path")
        _require(_read_json(target) == expected, f"stored preflight artifact changed: {relative}")
    validate_terminal_ledger(expected_ledger, expected_manifest)
    for forbidden in (ACCESS_INTENT_PATH, RESULT_PATH, ADJUDICATION_PATH, FAILURE_PATH):
        _require(
            not (repo / forbidden).exists(), f"production artifact already exists: {forbidden}"
        )
    return expected_preflight


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_committed_preflight(root: Path) -> str:
    package_paths = [
        CONFIG_PATH,
        MODULE_PATH,
        TEST_PATH,
        MANIFEST_PATH,
        TERMINAL_LEDGER_PATH,
        PREFLIGHT_PATH,
    ]
    for relative in package_paths:
        listed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative.as_posix()],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        _require(listed.returncode == 0, f"campaign preflight is not committed: {relative}")
        current = (root / relative).read_bytes()
        committed = _git_show(root, "HEAD", relative.as_posix())
        _require(current == committed, f"campaign preflight differs from HEAD: {relative}")
    return _git_head(root)


def build_access_intent(root: Path, package_commit: str) -> dict[str, Any]:
    preflight = build_preflight(root)
    _require(
        _read_json(root / PREFLIGHT_PATH) == preflight,
        "stored preflight changed before access intent",
    )
    manifest = _read_json(root / MANIFEST_PATH)
    ledger = _read_json(root / TERMINAL_LEDGER_PATH)
    validate_terminal_ledger(ledger, manifest)
    intent: dict[str, Any] = {
        "schema_version": ACCESS_INTENT_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "package_commit": package_commit,
        "manifest_file_sha256": file_sha256(root / MANIFEST_PATH),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "terminal_ledger_file_sha256": file_sha256(root / TERMINAL_LEDGER_PATH),
        "terminal_ledger_content_sha256": ledger["ledger_content_sha256"],
        "preflight_file_sha256": file_sha256(root / PREFLIGHT_PATH),
        "preflight_content_sha256": preflight["preflight_content_sha256"],
        "response_scope": {
            "SPARC_development_objects": 139,
            "SPARC_scored_rows": 2720,
            "XCOP_development_clusters": [
                "A1644",
                "A1795",
                "A2142",
                "A2255",
                "A2319",
                "A3266",
                "A85",
                "ZW1215",
            ],
            "confirmation_rows": 0,
            "independent_rows": 0,
            "group_rows": 0,
            "lensing_rows": 0,
        },
        "candidate_count": 407,
        "parameter_cell_count": 2486,
        "planned_scored_cells": {"GALAXIES": 179, "CLUSTERS": 1669},
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "external_cost_usd": 0.0,
        "replay_allowed": False,
        "intent_content_sha256": "",
    }
    intent["intent_content_sha256"] = _self_hash(intent, "intent_content_sha256")
    return intent


def _loss_rows(
    predictions: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_fractional_error: float,
) -> dict[str, Any]:
    by_observable: dict[str, list[float]] = {}
    per_row: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["row_id"])
        observed = float(row["observed"])
        error = float(row["error"])
        predicted = float(predictions[row_id])
        _require(observed > 0.0 and error >= 0.0, f"invalid response row: {row_id}")
        _require(math.isfinite(predicted) and predicted > 0.0, f"invalid prediction: {row_id}")
        fractional = max(error / observed, minimum_fractional_error)
        residual = math.log(predicted / observed) / fractional
        square = residual * residual
        observable = str(row["observable"])
        by_observable.setdefault(observable, []).append(square)
        per_row.append(
            {
                "row_id": row_id,
                "radius": float(row["radius_kpc"]),
                "observable": observable,
                "observed": observed,
                "error": error,
                "predicted": predicted,
                "standardized_residual": residual,
                "standardized_square": square,
            }
        )
    group_loss = {key: float(np.mean(values)) for key, values in by_observable.items()}
    loss = float(np.mean(list(group_loss.values())))
    worst = max(per_row, key=lambda row: (float(row["standardized_square"]), str(row["row_id"])))
    return {
        "loss": loss,
        "by_observable": group_loss,
        "row_count": len(rows),
        "worst_row_id": worst["row_id"],
        "worst_radius": worst["radius"],
        "worst_standardized_square": worst["standardized_square"],
        "per_row": per_row,
    }


def _sparc_source(
    galaxy: Any,
    nuisance: Mapping[str, Any],
) -> dict[str, Any]:
    disk_ml = Fraction(str(nuisance["disk_ml"]))
    bulge_ml = Fraction(str(nuisance["bulge_ml"]))
    radius_kpc = np.asarray([float(value) for value in galaxy.radius], dtype=float)
    radius_m = radius_kpc * 3.085677581491367e19
    signed_gas_v2 = np.asarray([float(value * abs(value)) for value in galaxy.v_gas], dtype=float)
    stellar_v2 = float(disk_ml) * np.square(
        np.asarray([float(value) for value in galaxy.v_disk])
    ) + float(bulge_ml) * np.square(np.asarray([float(value) for value in galaxy.v_bul]))
    vbar2 = signed_gas_v2 + stellar_v2
    _require(bool(np.all(vbar2 > 0.0)), f"nonpositive SPARC baryonic source: {galaxy.name}")
    gas_v2 = np.maximum(signed_gas_v2, 0.0)
    residual_v2 = vbar2 - gas_v2
    _require(bool(np.all(residual_v2 >= 0.0)), f"bad nonnegative source split: {galaxy.name}")
    gas_acceleration = gas_v2 * 1.0e6 / radius_m
    residual_acceleration = residual_v2 * 1.0e6 / radius_m
    bundle = adapter.compile_sparc_source_drivers(radius_m, gas_acceleration, residual_acceleration)
    return {
        "bundle": bundle,
        "radius_kpc": radius_kpc,
        "radius_m": radius_m,
        "vbar2": vbar2,
        "vobs": np.asarray([float(value) for value in galaxy.v_obs], dtype=float),
        "sigma": np.asarray([float(value) for value in galaxy.e_v_obs], dtype=float),
        "rows": galaxy.count,
        "nuisance_id": nuisance["cell_id"],
    }


def _sparc_score(prediction: np.ndarray, source: Mapping[str, Any]) -> dict[str, Any]:
    observed = np.asarray(source["vobs"], dtype=float)
    sigma = np.asarray(source["sigma"], dtype=float)
    _require(prediction.shape == observed.shape == sigma.shape, "SPARC score shape changed")
    standardized = (prediction - observed) / sigma
    square = standardized * standardized
    worst_index = int(np.argmax(square))
    return {
        "loss": float(np.mean(square)),
        "row_count": int(square.size),
        "worst_row_index": worst_index,
        "worst_radius": float(np.asarray(source["radius_kpc"])[worst_index]),
        "worst_standardized_square": float(square[worst_index]),
        "prediction": prediction,
        "standardized_residual": standardized,
    }


def _static_factor(
    concept: Mapping[str, Any],
    cell: Mapping[str, Any],
    bundle: Mapping[str, Any],
    domain: str,
) -> np.ndarray:
    parameters = cell["exact_value_or_rule"]["value"]
    if concept["entry_kind"] == "ATOMIC":
        driver_id = str(concept["driver_ids"][0])
        compiled = adapter.compile_static_architecture(
            str(concept["architecture_id"]),
            bundle["xi"],
            bundle["normalized"][driver_id],
            bundle["physical"]["D01_ACC"],
            parameters,
        )
    else:
        compiled = adapter.compile_compound_static(
            domain,
            str(concept["concept_id"]),
            bundle,
            parameters,
        )
    return np.asarray(compiled["primary"]["factor"], dtype=float)


def _gp01_l_factor(bundle: Mapping[str, Any], n: int) -> np.ndarray:
    g_b = np.asarray(bundle["physical"]["D01_ACC"], dtype=float)
    effective = adapter.gp01_l_acceleration(g_b, n=n)
    return np.divide(effective, g_b, out=np.ones_like(g_b), where=g_b > 0.0)


def _gp01_elliptic_factor(
    bundle: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    radius = np.asarray(bundle["radius_m"], dtype=float)
    mass = np.asarray(bundle["mass"]["baryonic_enclosed_kg"], dtype=float)
    rho = np.asarray(bundle["physical"]["D04_RHO"], dtype=float)
    tide = np.asarray(bundle["physical"]["D07_TIDE"], dtype=float)
    weight = adapter.gp01_environment_weight(
        rho,
        tide,
        rho_star=float(parameters["rho_ratio"]) * float(bundle["metadata"]["rho_reference_kg_m3"]),
        tide_star=float(parameters["tide_ratio"])
        * float(bundle["metadata"]["tidal_reference_s_minus_2"]),
        q=int(parameters["q"]),
        tide_power=int(parameters["tide_power"]),
    )
    target = adapter.gp01_bounded_target(
        bundle["physical"]["D01_ACC"],
        weight,
        n=int(parameters["n"]),
        A_max=float(parameters["A_max"]),
    )
    solved = adapter.solve_spherical_gamma(
        radius,
        target,
        L_g_m=float(parameters["L_ratio"]) * float(bundle["metadata"]["R_b_m"]),
    )
    flux = adapter.integrated_spherical_flux(radius, mass, solved["gamma"])
    _require(solved["finite"] and solved["positive"], "GP01 real-source solve failed")
    _require(solved["operator_residual_max_abs"] <= 1.0e-9, "GP01 operator gate failed")
    _require(solved["boundary_residual_max_abs"] <= 1.0e-10, "GP01 boundary gate failed")
    _require(
        flux["integrated_flux_relative_residual"] <= 1.0e-12,
        "GP01 flux gate failed",
    )
    return np.asarray(flux["factor"], dtype=float), {
        "operator_residual": float(solved["operator_residual_max_abs"]),
        "boundary_residual": float(solved["boundary_residual_max_abs"]),
        "flux_residual": float(flux["integrated_flux_relative_residual"]),
    }


def _scenario_rows(config: Mapping[str, Any], domain: str) -> list[dict[str, Any]]:
    return [dict(row) for row in config["nuisance_cases"] if row["domain"] == domain]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        _require(math.isfinite(value), "nonfinite result value")
    return value


def _eligible_cells(
    manifest: Mapping[str, Any], domain: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates = {str(row["candidate_id"]): row for row in manifest["candidate_versions"]}
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cell in manifest["parameter_cells"]:
        rule = cell["exact_value_or_rule"]
        concept_id = str(rule["concept_id"])
        candidate = candidates[concept_id]
        disposition = candidate["domain_execution"][domain]
        source = rule["domain_sources"][domain]
        if disposition["eligible"] and str(source["readiness"]).startswith("SOURCE_READY_"):
            rows.append((dict(candidate), dict(cell)))
    rows.sort(key=lambda item: str(item[1]["cell_id"]))
    expected = {"GALAXIES": 179, "CLUSTERS": 1669}[domain]
    _require(len(rows) == expected, f"eligible {domain} cell count changed")
    return rows


def _require_git_payload_hash(repo: Path, commit: str, relative: str, expected_sha256: str) -> None:
    try:
        committed_payload = _git_show(repo, commit, relative)
    except subprocess.CalledProcessError as error:
        raise OpenGravityCampaignError(f"committed scientific input missing: {relative}") from error
    _require(
        hashlib.sha256(committed_payload).hexdigest() == expected_sha256,
        f"committed scientific input bytes changed: {relative}",
    )


def _verify_scientific_input_contracts(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify immutable input identities without opening a scientific response payload."""

    from sigma_theory_compiler import gravity_extended_source_clock_xcop_development as clock
    from sigma_theory_compiler import sparc_full_sample

    repo = root.resolve()
    _require(repo == _repo_root(), "only canonical scientific input root allowed")
    scientific = config["scientific_input_bindings"]
    sparc = scientific["sparc"]
    sparc_relative = str(sparc["dataset_path"])
    _require(sparc_relative == sparc_full_sample.DATASET_PATH, "SPARC loader path changed")
    sparc_path = (repo / sparc_relative).resolve()
    _require(
        sparc_path == (repo / "configs/sparc_rotation_curves_full_v1.json").resolve(),
        "SPARC dataset path escaped the frozen contract",
    )
    _require_git_payload_hash(
        repo,
        str(sparc["dataset_commit"]),
        sparc_relative,
        str(sparc["dataset_raw_sha256"]),
    )

    clock_config = clock.load_config(repo)
    input_contract = clock_config.get("input_contract")
    _require(isinstance(input_contract, Mapping), "X-COP input contract missing")
    xcop = scientific["xcop"]
    _require(
        content_sha256(input_contract) == xcop["input_contract_sha256"],
        "X-COP input contract changed",
    )
    _require(input_contract.get("raw_root") == xcop["raw_root"], "X-COP raw root changed")
    files = input_contract.get("files")
    _require(isinstance(files, list), "X-COP input file ledger missing")
    _require(
        len(files) == int(input_contract["unique_files"]) == int(xcop["unique_files"]) == 29,
        "X-COP input file count changed",
    )
    _require(
        sum(int(row["bytes"]) for row in files)
        == int(input_contract["total_bytes"])
        == int(xcop["total_bytes"])
        == 538560,
        "X-COP input byte scope changed",
    )
    allowed_clusters = list(config["data_contract"]["xcop_objects"])
    stellar_clusters = set(config["data_contract"]["xcop_stellar_available"])
    expected_roles = set(xcop["required_roles"])
    _require(
        expected_roles == {"density", "pressure", "stellar_mass", "temperature"},
        "X-COP role vocabulary changed",
    )
    raw_root = (repo / str(input_contract["raw_root"])).resolve()
    expected_raw_root = (
        repo / "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw"
    ).resolve()
    _require(raw_root == expected_raw_root, "X-COP raw root escaped the frozen contract")
    by_cluster: dict[str, set[str]] = {cluster: set() for cluster in allowed_clusters}
    members: set[str] = set()
    for row in files:
        _require(
            set(row) == {"cluster", "role", "member", "bytes", "sha256"},
            "X-COP file row schema changed",
        )
        cluster = str(row["cluster"])
        role = str(row["role"])
        member = str(row["member"])
        _require(cluster in by_cluster, "X-COP file escaped the cluster allowlist")
        _require(role in expected_roles, "X-COP file role changed")
        _require(member not in members, "duplicate X-COP input member")
        members.add(member)
        _require("\x00" not in member and "\\" not in member, "unsafe X-COP member path")
        member_path = Path(member)
        _require(not member_path.is_absolute(), "absolute X-COP member path")
        _require(".." not in member_path.parts, "traversing X-COP member path")
        _require(
            (raw_root / member_path).resolve().is_relative_to(raw_root), "escaped X-COP member path"
        )
        _require(type(row["bytes"]) is int and int(row["bytes"]) > 0, "invalid X-COP byte seal")
        _require(_SHA256_RE.fullmatch(str(row["sha256"])) is not None, "invalid X-COP hash seal")
        by_cluster[cluster].add(role)
    for cluster, roles in by_cluster.items():
        required = {"density", "pressure", "temperature"}
        if cluster in stellar_clusters:
            required.add("stellar_mass")
        _require(roles == required, f"X-COP role inventory changed: {cluster}")
    return dict(clock_config)


def _load_exact_sparc_dataset(path: Path, expected_sha256: str) -> tuple[list[Any], dict[str, Any]]:
    """Open the SPARC container once, seal its raw bytes, then decode its rows."""

    from sigma_theory_compiler import sparc_full_sample

    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "SPARC dataset raw bytes changed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenGravityCampaignError("SPARC dataset encoding or JSON changed") from error
    _require(isinstance(payload, Mapping), "SPARC dataset object changed")
    sparc_full_sample.validate_dataset(payload)
    galaxies: list[Any] = []
    for entry in payload["galaxies"]:
        columns = list(zip(*entry["rows"], strict=True))
        galaxies.append(
            sparc_full_sample.Galaxy(
                name=entry["name"],
                distance_mpc=entry["distance_mpc"],
                radius=tuple(sparc_full_sample._decimal(value) for value in columns[0]),
                v_obs=tuple(sparc_full_sample._decimal(value) for value in columns[1]),
                e_v_obs=tuple(sparc_full_sample._decimal(value) for value in columns[2]),
                v_gas=tuple(sparc_full_sample._decimal(value) for value in columns[3]),
                v_disk=tuple(sparc_full_sample._decimal(value) for value in columns[4]),
                v_bul=tuple(sparc_full_sample._decimal(value) for value in columns[5]),
                published=tuple(tuple(row) for row in entry["rows"]),
            )
        )
    provenance = {
        "columns": payload["columns"],
        "dataset_sha256": sparc_full_sample.canonical_sha256(payload),
        "galaxy_count": len(galaxies),
        "galaxy_digest_sha256": payload["galaxy_digest_sha256"],
        "mass_to_light_convention": payload["mass_to_light_convention"],
        "per_galaxy_provenance": {
            entry["name"]: dict(entry["provenance"]) for entry in payload["galaxies"]
        },
        "point_count": sum(item.count for item in galaxies),
        "selection": payload["selection"],
        "source": payload["source"],
        "raw_file_sha256": expected_sha256,
    }
    return galaxies, provenance


def _load_sparc_responses(
    root: Path, context: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    binding = config["scientific_input_bindings"]["sparc"]
    path = (root / str(binding["dataset_path"])).resolve()
    _require(
        path == (root / "configs/sparc_rotation_curves_full_v1.json").resolve(),
        "SPARC response path changed",
    )
    galaxies, provenance = _load_exact_sparc_dataset(path, str(binding["dataset_raw_sha256"]))
    admitted_names = set(map(str, context["source_predecessor"]["objects"]["SPARC"]))
    selected = sorted(
        (galaxy for galaxy in galaxies if galaxy.name in admitted_names),
        key=lambda galaxy: galaxy.name,
    )
    _require(
        len(galaxies) == 175 and sum(galaxy.count for galaxy in galaxies) == 3391,
        "SPARC parse scope changed",
    )
    _require(
        len(selected) == 139 and sum(galaxy.count for galaxy in selected) == 2720,
        "SPARC score scope changed",
    )
    _require(
        {galaxy.name for galaxy in selected} == admitted_names, "SPARC admitted ledger changed"
    )
    return selected, provenance


def _cluster_nuisance(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "outer_nonthermal_fraction": float(row["outer_nonthermal_fraction"]),
        "published_stellar_mass_scale": float(row["published_stellar_mass_scale"]),
        "missing_stellar_to_gas_mass_ratio": float(row["missing_stellar_to_gas_mass_ratio"]),
        "xray_temperature_cross_calibration": float(row["xray_temperature_cross_calibration"]),
        "nonthermal_radial_power": 1.0,
    }


def _load_xcop_responses(
    root: Path, config: Mapping[str, Any], clock_config: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from sigma_theory_compiler import gravity_extended_source_clock_xcop_development as clock
    from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

    bound_clock_config = (
        _verify_scientific_input_contracts(root, config)
        if clock_config is None
        else dict(clock_config)
    )
    _require(
        content_sha256(bound_clock_config["input_contract"])
        == config["scientific_input_bindings"]["xcop"]["input_contract_sha256"],
        "X-COP response input contract changed",
    )
    item59_config = item59.load_config(root)
    payloads, file_ledger = clock._load_allowed_payloads(root, bound_clock_config)
    packets: list[dict[str, Any]] = []
    for cluster in config["data_contract"]["xcop_objects"]:
        packet = clock._parse_packet(str(cluster), payloads, item59_config)
        clock._add_rows(packet, item59_config)
        packets.append(packet)
    _require(len(packets) == 8, "X-COP packet count changed")
    _require(sum(len(packet["rows"]) for packet in packets) > 0, "X-COP responses are empty")
    return packets, file_ledger, bound_clock_config, item59_config


def _cluster_state_bundle(
    packet: Mapping[str, Any], scenario: Mapping[str, Any], item59_config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray]:
    from sigma_theory_compiler import gravity_cluster_comparator_suite as cluster_suite

    scaled_packet = copy.deepcopy(dict(packet))
    scaled_packet["ne_cm3"] = np.asarray(packet["ne_cm3"], dtype=float) * float(
        scenario["density_scale"]
    )
    nuisance = _cluster_nuisance(scenario)
    state = cluster_suite._state(scaled_packet, nuisance, item59_config)
    constants = item59_config["constants"]
    rho = (
        np.asarray(state["calc_ne"], dtype=float)
        * 1.0e6
        * float(constants["mean_molecular_weight_per_electron"])
        * float(constants["proton_mass_kg"])
    )
    stellar = None
    if packet.get("stellar") is not None:
        stellar = np.asarray(state["member_mass"], dtype=float)
    bundle = adapter.compile_xcop_spherical_source_drivers(
        state["radius_m"],
        rho,
        stellar,
        missing_stellar_to_gas_ratio=0.1,
    )
    gbar = (
        float(constants["gravity_si"])
        * (np.asarray(state["gas_mass"]) + np.asarray(state["member_mass"]))
        / np.maximum(np.asarray(state["radius_m"]) ** 2, np.finfo(float).tiny)
    )
    return scaled_packet, state, bundle, gbar


def _factor_on_radii(
    factor: Sequence[float] | np.ndarray,
    bundle: Mapping[str, Any],
    target_radius_m: Sequence[float] | np.ndarray,
) -> np.ndarray:
    result = np.interp(
        np.asarray(target_radius_m, dtype=float),
        np.asarray(bundle["radius_m"], dtype=float),
        np.asarray(factor, dtype=float),
    )
    _require(bool(np.all(np.isfinite(result) & (result > 0.0))), "invalid radial factor")
    return result


def _factor_for_cell(
    candidate: Mapping[str, Any],
    cell: Mapping[str, Any],
    concept_packet: Mapping[str, Mapping[str, Any]],
    bundle: Mapping[str, Any],
    domain: str,
) -> tuple[np.ndarray, dict[str, float]]:
    concept_id = str(candidate["candidate_id"])
    if concept_id == "GP01-L":
        parameters = cell["exact_value_or_rule"]["value"]
        return _gp01_l_factor(bundle, int(parameters["n"])), {}
    if concept_id == "GP01-ELLIPTIC":
        return _gp01_elliptic_factor(bundle, cell["exact_value_or_rule"]["value"])
    concept = concept_packet[concept_id]
    factor = _static_factor(concept, cell, bundle, domain)
    return factor, {}


def _aggregate_object_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    losses = [float(row["loss"]) for row in rows]
    _require(bool(losses), "empty object-score aggregation")
    worst = max(rows, key=lambda row: (float(row["loss"]), str(row["object"])))
    return {
        "mean_loss": float(np.mean(losses)),
        "object_count": len(rows),
        "worst_object": str(worst["object"]),
        "worst_object_loss": float(worst["loss"]),
    }


def _score_sparc_comparators(
    galaxies: Sequence[Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], float]]:
    from sigma_theory_compiler import gravity_g0_experiment as g0

    rows: list[dict[str, Any]] = []
    object_losses: dict[tuple[str, str, str], float] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["cell_id"])
        for galaxy in galaxies:
            source = _sparc_source(galaxy, scenario)
            bundle = source["bundle"]
            gbar = np.asarray(bundle["physical"]["D01_ACC"], dtype=float)
            rar = np.ones_like(gbar)
            positive = gbar > 0.0
            rar[positive] = 1.0 / -np.expm1(-np.sqrt(gbar[positive] / 1.2e-10))
            factor_map = {
                "BARYON_ONLY": np.ones_like(gbar),
                "EMPIRICAL_RAR": rar,
                "ALGEBRAIC_MOND_GP01_L_n1": _gp01_l_factor(bundle, 1),
                "ALGEBRAIC_MOND_GP01_L_n2": _gp01_l_factor(bundle, 2),
                "ALGEBRAIC_MOND_GP01_L_n4": _gp01_l_factor(bundle, 4),
            }
            for comparator_id, grid_factor in factor_map.items():
                factor = _factor_on_radii(grid_factor, bundle, source["radius_m"])
                score = _sparc_score(np.sqrt(factor * source["vbar2"]), source)
                loss = float(score["loss"])
                object_losses[(comparator_id, scenario_id, galaxy.name)] = loss
                rows.append(
                    {
                        "comparator_id": comparator_id,
                        "scenario_id": scenario_id,
                        "object": galaxy.name,
                        "loss": loss,
                        "worst_radius": score["worst_radius"],
                    }
                )

            arrays = {
                "radius": np.asarray(source["radius_kpc"], dtype=float),
                "vbar2": np.asarray(source["vbar2"], dtype=float),
                "vobs": np.asarray(source["vobs"], dtype=float),
                "sigma": np.asarray(source["sigma"], dtype=float),
            }
            indices = tuple(range(galaxy.count))
            amplitude, scale, _fit_loss = g0._fit_nfw_fold(arrays, indices, 41)
            prediction = np.sqrt(
                np.maximum(
                    arrays["vbar2"] + amplitude * g0._nfw_shape(arrays["radius"], scale),
                    0.0,
                )
            )
            score = _sparc_score(prediction, source)
            comparator_id = "GR_PLUS_NFW_CONTEXTUAL_CEILING"
            object_losses[(comparator_id, scenario_id, galaxy.name)] = float(score["loss"])
            rows.append(
                {
                    "comparator_id": comparator_id,
                    "scenario_id": scenario_id,
                    "object": galaxy.name,
                    "loss": float(score["loss"]),
                    "worst_radius": score["worst_radius"],
                    "fitted_response_parameters": {
                        "amplitude": amplitude,
                        "scale_radius_kpc": scale,
                    },
                }
            )
    return rows, object_losses


def _score_xcop_comparators(
    packets: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    clock_config: Mapping[str, Any],
    item59_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], float]]:
    from sigma_theory_compiler import gravity_cluster_comparator_suite as cluster_suite
    from sigma_theory_compiler import gravity_extended_source_clock_xcop_development as clock

    rows: list[dict[str, Any]] = []
    object_losses: dict[tuple[str, str, str], float] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["cell_id"])
        nuisance = _cluster_nuisance(scenario)
        scaled_packets = []
        for packet in packets:
            scaled = copy.deepcopy(dict(packet))
            scaled["ne_cm3"] = np.asarray(packet["ne_cm3"], dtype=float) * float(
                scenario["density_scale"]
            )
            scaled_packets.append(scaled)
        for law_id, comparator_id in (
            ("newtonian_baryons", "BARYON_ONLY"),
            ("empirical_rar", "EMPIRICAL_RAR"),
            ("previous_cross_scale_candidate", "PREVIOUS_CROSS_SCALE"),
            ("extended_source_clock", "EXTENDED_SOURCE_CLOCK"),
        ):
            for packet in scaled_packets:
                predictions, _diagnostics = clock._predict_law(
                    packet, law_id, item59_config, nuisance
                )
                score = _loss_rows(
                    predictions,
                    packet["rows"],
                    minimum_fractional_error=0.05,
                )
                loss = float(score["loss"])
                object_losses[(comparator_id, scenario_id, str(packet["cluster"]))] = loss
                rows.append(
                    {
                        "comparator_id": comparator_id,
                        "scenario_id": scenario_id,
                        "object": str(packet["cluster"]),
                        "loss": loss,
                        "by_observable": score["by_observable"],
                        "worst_radius": score["worst_radius"],
                    }
                )

        for n in (1, 2, 4):
            comparator_id = f"ALGEBRAIC_MOND_GP01_L_n{n}"
            for packet in scaled_packets:
                scaled, state, bundle, gbar = _cluster_state_bundle(
                    packet, {**scenario, "density_scale": 1.0}, item59_config
                )
                grid_factor = _gp01_l_factor(bundle, n)
                factor = _factor_on_radii(grid_factor, bundle, state["radius_m"])
                predictions = cluster_suite._predictions_from_acceleration(
                    scaled, state, factor * gbar, nuisance, item59_config
                )
                score = _loss_rows(predictions, scaled["rows"], minimum_fractional_error=0.05)
                loss = float(score["loss"])
                object_losses[(comparator_id, scenario_id, str(packet["cluster"]))] = loss
                rows.append(
                    {
                        "comparator_id": comparator_id,
                        "scenario_id": scenario_id,
                        "object": str(packet["cluster"]),
                        "loss": loss,
                        "by_observable": score["by_observable"],
                        "worst_radius": score["worst_radius"],
                    }
                )

        suite_config = cluster_suite.load_config(_repo_root())
        for model in suite_config["parametric_gravity_models"][:2]:
            model_id = str(model["model_id"])
            best: tuple[float, str, dict[str, float], dict[str, float]] | None = None
            for parameters in cluster_suite._parameter_rows(model):
                predictions = cluster_suite._gravity_model_predictions(
                    scaled_packets, model_id, parameters, nuisance, item59_config
                )
                per_object = {}
                for packet in scaled_packets:
                    score = _loss_rows(
                        predictions,
                        packet["rows"],
                        minimum_fractional_error=0.05,
                    )
                    per_object[str(packet["cluster"])] = float(score["loss"])
                mean = float(np.mean(list(per_object.values())))
                tie = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
                row = (mean, tie, dict(parameters), per_object)
                if best is None or row[:2] < best[:2]:
                    best = row
            _require(best is not None, f"empty cluster comparator grid: {model_id}")
            _, _, parameters, per_object = best
            comparator_id = f"{model_id}_CONTEXTUAL_CEILING"
            for cluster, loss in sorted(per_object.items()):
                object_losses[(comparator_id, scenario_id, cluster)] = loss
                rows.append(
                    {
                        "comparator_id": comparator_id,
                        "scenario_id": scenario_id,
                        "object": cluster,
                        "loss": loss,
                        "fitted_response_parameters": parameters,
                    }
                )
    return rows, object_losses


def _score_sparc_candidates(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    galaxies: Sequence[Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    concept_packet = {str(row["concept_id"]): row for row in context["packet"]["twell_rows"]}
    source_cache = {
        (galaxy.name, str(scenario["cell_id"])): _sparc_source(galaxy, scenario)
        for scenario in scenarios
        for galaxy in galaxies
    }
    results: list[dict[str, Any]] = []
    for candidate, cell in _eligible_cells(manifest, "GALAXIES"):
        scenario_rows = []
        for scenario in scenarios:
            scenario_id = str(scenario["cell_id"])
            object_rows = []
            for galaxy in galaxies:
                source = source_cache[(galaxy.name, scenario_id)]
                grid_factor, diagnostics = _factor_for_cell(
                    candidate,
                    cell,
                    concept_packet,
                    source["bundle"],
                    "SPARC",
                )
                factor = _factor_on_radii(grid_factor, source["bundle"], source["radius_m"])
                reversed_factor = _factor_on_radii(
                    np.asarray(grid_factor)[::-1],
                    source["bundle"],
                    source["radius_m"],
                )
                score = _sparc_score(np.sqrt(factor * np.asarray(source["vbar2"])), source)
                reverse_score = _sparc_score(
                    np.sqrt(reversed_factor * np.asarray(source["vbar2"])), source
                )
                object_rows.append(
                    {
                        "object": galaxy.name,
                        "loss": float(score["loss"]),
                        "reversal_loss": float(reverse_score["loss"]),
                        "row_count": int(score["row_count"]),
                        "worst_radius": float(score["worst_radius"]),
                        "worst_standardized_square": float(score["worst_standardized_square"]),
                        "operator_diagnostics": diagnostics,
                    }
                )
            summary = _aggregate_object_scores(object_rows)
            summary.update(
                {
                    "scenario_id": scenario_id,
                    "reversal_mean_loss": float(
                        np.mean([row["reversal_loss"] for row in object_rows])
                    ),
                    "objects": object_rows,
                }
            )
            scenario_rows.append(summary)
        results.append(
            {
                "cell_id": str(cell["cell_id"]),
                "concept_id": str(candidate["candidate_id"]),
                "anonymous_formula_id": str(candidate["anonymous_formula_id"]),
                "lane": str(candidate["lane"]),
                "domain": "GALAXIES",
                "robust_loss": max(float(row["mean_loss"]) for row in scenario_rows),
                "scenario_results": scenario_rows,
            }
        )
    return results


def _score_xcop_candidates(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    item59_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from sigma_theory_compiler import gravity_cluster_comparator_suite as cluster_suite

    concept_packet = {str(row["concept_id"]): row for row in context["packet"]["twell_rows"]}
    source_cache = {}
    for scenario in scenarios:
        for packet in packets:
            source_cache[(str(packet["cluster"]), str(scenario["cell_id"]))] = (
                _cluster_state_bundle(packet, scenario, item59_config)
            )
    results: list[dict[str, Any]] = []
    for candidate, cell in _eligible_cells(manifest, "CLUSTERS"):
        scenario_rows = []
        for scenario in scenarios:
            scenario_id = str(scenario["cell_id"])
            nuisance = _cluster_nuisance(scenario)
            object_rows = []
            for packet in packets:
                cluster = str(packet["cluster"])
                scaled, state, bundle, gbar = source_cache[(cluster, scenario_id)]
                grid_factor, diagnostics = _factor_for_cell(
                    candidate,
                    cell,
                    concept_packet,
                    bundle,
                    "XCOP_SPHERICAL",
                )
                factor = _factor_on_radii(grid_factor, bundle, state["radius_m"])
                reverse_factor = _factor_on_radii(
                    np.asarray(grid_factor)[::-1], bundle, state["radius_m"]
                )
                predictions = cluster_suite._predictions_from_acceleration(
                    scaled, state, factor * gbar, nuisance, item59_config
                )
                reversed_predictions = cluster_suite._predictions_from_acceleration(
                    scaled, state, reverse_factor * gbar, nuisance, item59_config
                )
                score = _loss_rows(predictions, scaled["rows"], minimum_fractional_error=0.05)
                reverse_score = _loss_rows(
                    reversed_predictions,
                    scaled["rows"],
                    minimum_fractional_error=0.05,
                )
                object_rows.append(
                    {
                        "object": cluster,
                        "loss": float(score["loss"]),
                        "reversal_loss": float(reverse_score["loss"]),
                        "by_observable": score["by_observable"],
                        "row_count": int(score["row_count"]),
                        "worst_radius": float(score["worst_radius"]),
                        "worst_standardized_square": float(score["worst_standardized_square"]),
                        "operator_diagnostics": diagnostics,
                    }
                )
            summary = _aggregate_object_scores(object_rows)
            summary.update(
                {
                    "scenario_id": scenario_id,
                    "reversal_mean_loss": float(
                        np.mean([row["reversal_loss"] for row in object_rows])
                    ),
                    "objects": object_rows,
                }
            )
            scenario_rows.append(summary)
        results.append(
            {
                "cell_id": str(cell["cell_id"]),
                "concept_id": str(candidate["candidate_id"]),
                "anonymous_formula_id": str(candidate["anonymous_formula_id"]),
                "lane": str(candidate["lane"]),
                "domain": "CLUSTERS",
                "robust_loss": max(float(row["mean_loss"]) for row in scenario_rows),
                "scenario_results": scenario_rows,
            }
        )
    return results


def _comparator_summary(rows: Sequence[Mapping[str, Any]], domain: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["comparator_id"]), str(row["scenario_id"])), []).append(row)
    summaries = []
    for (comparator_id, scenario_id), object_rows in sorted(grouped.items()):
        summary = _aggregate_object_scores(object_rows)
        summary.update(
            {
                "comparator_id": comparator_id,
                "scenario_id": scenario_id,
                "objects": [
                    {"object": str(row["object"]), "loss": float(row["loss"])}
                    for row in sorted(object_rows, key=lambda item: str(item["object"]))
                ],
            }
        )
        summaries.append(summary)
    return {"domain": domain, "scenario_results": summaries}


def _strongest_comparator_by_scenario(
    comparator_summary: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in comparator_summary["scenario_results"]:
        grouped.setdefault(str(row["scenario_id"]), []).append(row)
    return {
        scenario: min(
            rows,
            key=lambda row: (float(row["mean_loss"]), str(row["comparator_id"])),
        )
        for scenario, rows in grouped.items()
    }


def _subset_names(
    context: Mapping[str, Any], config: Mapping[str, Any], domain: str
) -> tuple[set[str], set[str]]:
    if domain == "GALAXIES":
        all_names = set(map(str, context["source_predecessor"]["objects"]["SPARC"]))
        pilot = set(map(str, context["source_predecessor"]["partition_design"]["SPARC_pilot"]))
        return pilot, all_names - pilot
    pilot = set(map(str, config["data_contract"]["xcop_pilot"]))
    return pilot, set(map(str, config["data_contract"]["xcop_full"]))


def _adjudicate_domain(
    domain: str,
    candidate_results: Sequence[Mapping[str, Any]],
    comparator_summary: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    strongest = _strongest_comparator_by_scenario(comparator_summary)
    pilot, full = _subset_names(context, config, domain)
    minimum_support = (
        int(config["metrics"]["minimum_galaxy_support"])
        if domain == "GALAXIES"
        else int(config["metrics"]["minimum_cluster_support"])
    )
    threshold = float(config["metrics"]["minimum_meaningful_improvement"])
    rows = []
    for candidate in candidate_results:
        scenario_evidence = []
        supported: set[str] | None = None
        loo_min = math.inf
        worst_ratio = 0.0
        subgroup_ratio = 0.0
        for scenario in candidate["scenario_results"]:
            scenario_id = str(scenario["scenario_id"])
            control = strongest[scenario_id]
            control_map = {str(row["object"]): float(row["loss"]) for row in control["objects"]}
            candidate_map = {str(row["object"]): float(row["loss"]) for row in scenario["objects"]}
            names = sorted(candidate_map)
            _require(set(names) == set(control_map), "candidate/comparator object ledger differs")
            scenario_supported = {
                name
                for name in names
                if candidate_map[name]
                < (1.0 - threshold) * max(control_map[name], np.finfo(float).tiny)
            }
            supported = scenario_supported if supported is None else supported & scenario_supported
            control_mean = float(control["mean_loss"])
            candidate_mean = float(scenario["mean_loss"])
            improvement = (control_mean - candidate_mean) / max(control_mean, np.finfo(float).tiny)
            ratios = [
                candidate_map[name] / max(control_map[name], np.finfo(float).tiny) for name in names
            ]
            worst_ratio = max(worst_ratio, max(ratios))
            for removed in names:
                remaining = [name for name in names if name != removed]
                c_mean = float(np.mean([candidate_map[name] for name in remaining]))
                r_mean = float(np.mean([control_map[name] for name in remaining]))
                loo_min = min(
                    loo_min,
                    (r_mean - c_mean) / max(r_mean, np.finfo(float).tiny),
                )
            for subset in (pilot, full):
                c_mean = float(np.mean([candidate_map[name] for name in subset]))
                r_mean = float(np.mean([control_map[name] for name in subset]))
                subgroup_ratio = max(
                    subgroup_ratio,
                    c_mean / max(r_mean, np.finfo(float).tiny),
                )
            pilot_candidate_mean = float(np.mean([candidate_map[name] for name in pilot]))
            pilot_comparator_mean = float(np.mean([control_map[name] for name in pilot]))
            full_candidate_mean = float(np.mean([candidate_map[name] for name in full]))
            full_comparator_mean = float(np.mean([control_map[name] for name in full]))
            scenario_evidence.append(
                {
                    "scenario_id": scenario_id,
                    "strongest_comparator": control["comparator_id"],
                    "candidate_mean_loss": candidate_mean,
                    "comparator_mean_loss": control_mean,
                    "fractional_improvement": improvement,
                    "passes_two_percent": improvement >= threshold,
                    "pilot_stage": {
                        "candidate_mean_loss": pilot_candidate_mean,
                        "comparator_mean_loss": pilot_comparator_mean,
                        "formula_changes_after_stage": 0,
                        "partial_ranking_released_before_full": False,
                    },
                    "full_development_stage": {
                        "candidate_mean_loss": full_candidate_mean,
                        "comparator_mean_loss": full_comparator_mean,
                        "formula_version_unchanged_from_pilot": True,
                    },
                }
            )
        support = len(supported or set())
        gates = {
            "EVERY_NUISANCE_CASE": all(
                bool(row["passes_two_percent"]) for row in scenario_evidence
            ),
            "OBJECT_BREADTH": support >= minimum_support,
            "LEAVE_ONE_OBJECT_OUT": loo_min
            >= float(config["metrics"]["leave_one_object_out_minimum_improvement"]),
            "WORST_OBJECT": worst_ratio
            <= float(config["metrics"]["maximum_worst_object_loss_ratio"]),
            "PILOT_FULL_SUBGROUP": subgroup_ratio
            <= float(config["metrics"]["maximum_subgroup_loss_ratio"]),
        }
        rows.append(
            {
                "cell_id": str(candidate["cell_id"]),
                "concept_id": str(candidate["concept_id"]),
                "domain": domain,
                "passes": all(gates.values()),
                "gates": gates,
                "support_count": support,
                "minimum_loo_improvement": loo_min,
                "worst_object_loss_ratio": worst_ratio,
                "maximum_subgroup_loss_ratio": subgroup_ratio,
                "scenario_evidence": scenario_evidence,
                "counterexample": min(
                    (
                        row
                        for scenario in candidate["scenario_results"]
                        for row in scenario["objects"]
                    ),
                    key=lambda row: (-float(row["loss"]), str(row["object"])),
                ),
            }
        )
    return rows


def _cross_domain_adjudication(
    galaxy_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    galaxies = {str(row["cell_id"]): row for row in galaxy_rows}
    clusters = {str(row["cell_id"]): row for row in cluster_rows}
    rows = []
    for cell_id in sorted(set(galaxies) & set(clusters)):
        galaxy = galaxies[cell_id]
        cluster = clusters[cell_id]
        rows.append(
            {
                "cell_id": cell_id,
                "concept_id": galaxy["concept_id"],
                "galaxies_pass": bool(galaxy["passes"]),
                "clusters_pass": bool(cluster["passes"]),
                "cross_domain_pass": bool(galaxy["passes"] and cluster["passes"]),
                "combined_worst_fractional_improvement": min(
                    min(
                        float(row["fractional_improvement"]) for row in galaxy["scenario_evidence"]
                    ),
                    min(
                        float(row["fractional_improvement"]) for row in cluster["scenario_evidence"]
                    ),
                ),
            }
        )
    return rows


def _best_object_candidates(
    results: Sequence[Mapping[str, Any]], object_name: str, limit: int = 10
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        losses = []
        for scenario in result["scenario_results"]:
            match = next(row for row in scenario["objects"] if row["object"] == object_name)
            losses.append(float(match["loss"]))
        rows.append(
            {
                "cell_id": str(result["cell_id"]),
                "concept_id": str(result["concept_id"]),
                "worst_nuisance_loss": max(losses),
                "nuisance_losses": losses,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["worst_nuisance_loss"]),
            str(row["cell_id"]),
        )
    )
    return rows[:limit]


def _nominal_sparc_prediction(
    cell_id: str,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    galaxy: Any,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    cells = {str(row["cell_id"]): row for row in manifest["parameter_cells"]}
    candidates = {str(row["candidate_id"]): row for row in manifest["candidate_versions"]}
    cell = cells[cell_id]
    candidate = candidates[str(cell["exact_value_or_rule"]["concept_id"])]
    concepts = {str(row["concept_id"]): row for row in context["packet"]["twell_rows"]}
    source = _sparc_source(galaxy, scenario)
    grid_factor, _diagnostics = _factor_for_cell(
        candidate, cell, concepts, source["bundle"], "SPARC"
    )
    factor = _factor_on_radii(grid_factor, source["bundle"], source["radius_m"])
    prediction = np.sqrt(factor * np.asarray(source["vbar2"]))
    baryonic = _factor_on_radii(
        np.asarray(source["bundle"]["physical"]["D01_ACC"]),
        source["bundle"],
        source["radius_m"],
    )
    return {
        "source_profile": [
            {
                "radius_kpc": float(radius),
                "baryonic_acceleration_m_s2": float(acceleration),
            }
            for radius, acceleration in zip(source["radius_kpc"], baryonic, strict=True)
        ],
        "candidate_state": [
            {"radius_kpc": float(radius), "gain_factor": float(gain)}
            for radius, gain in zip(source["radius_kpc"], factor, strict=True)
        ],
        "radial_prediction": [
            {
                "radius_kpc": float(radius),
                "observed": float(observed),
                "error": float(error),
                "predicted": float(predicted),
            }
            for radius, observed, error, predicted in zip(
                source["radius_kpc"],
                source["vobs"],
                source["sigma"],
                prediction,
                strict=True,
            )
        ],
    }


def _nominal_xcop_prediction(
    cell_id: str,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    packet: Mapping[str, Any],
    scenario: Mapping[str, Any],
    item59_config: Mapping[str, Any],
) -> dict[str, Any]:
    from sigma_theory_compiler import gravity_cluster_comparator_suite as cluster_suite

    cells = {str(row["cell_id"]): row for row in manifest["parameter_cells"]}
    candidates = {str(row["candidate_id"]): row for row in manifest["candidate_versions"]}
    cell = cells[cell_id]
    candidate = candidates[str(cell["exact_value_or_rule"]["concept_id"])]
    concepts = {str(row["concept_id"]): row for row in context["packet"]["twell_rows"]}
    scaled, state, bundle, gbar = _cluster_state_bundle(packet, scenario, item59_config)
    grid_factor, _diagnostics = _factor_for_cell(
        candidate, cell, concepts, bundle, "XCOP_SPHERICAL"
    )
    factor = _factor_on_radii(grid_factor, bundle, state["radius_m"])
    predictions = cluster_suite._predictions_from_acceleration(
        scaled,
        state,
        factor * gbar,
        _cluster_nuisance(scenario),
        item59_config,
    )
    return {
        "source_profile": [
            {
                "radius_kpc": float(radius / 3.085677581491367e19),
                "baryonic_acceleration_m_s2": float(acceleration),
                "gas_density_kg_m3": float(density),
                "baryonic_enclosed_mass_kg": float(mass),
            }
            for radius, acceleration, density, mass in zip(
                state["radius_m"],
                gbar,
                np.interp(state["radius_m"], bundle["radius_m"], bundle["physical"]["D04_RHO"]),
                np.asarray(state["gas_mass"], dtype=float)
                + np.asarray(state["member_mass"], dtype=float),
                strict=True,
            )
        ],
        "candidate_state": [
            {
                "radius_kpc": float(radius / 3.085677581491367e19),
                "gain_factor": float(gain),
            }
            for radius, gain in zip(state["radius_m"], factor, strict=True)
        ],
        "radial_prediction": [
            {
                "row_id": str(row["row_id"]),
                "observable": str(row["observable"]),
                "radius_kpc": float(row["radius_kpc"]),
                "observed": float(row["observed"]),
                "error": float(row["error"]),
                "predicted": float(predictions[str(row["row_id"])]),
            }
            for row in scaled["rows"]
        ],
    }


def _object_status(
    top_candidates: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
) -> str:
    if not top_candidates or not comparator_rows:
        return "SOURCE_BLOCKED"
    candidate = float(top_candidates[0]["worst_nuisance_loss"])
    by_scenario: dict[str, list[float]] = {}
    for row in comparator_rows:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(float(row["loss"]))
    comparator = max(min(losses) for losses in by_scenario.values())
    ratio = candidate / max(comparator, np.finfo(float).tiny)
    if ratio <= 0.98:
        return "SUPPORTS"
    if ratio >= 1.02:
        return "COUNTEREXAMPLE"
    return "NEUTRAL"


def _normalized_dashboard_comparators(
    rows: Sequence[Mapping[str, Any]], object_name: str
) -> list[dict[str, Any]]:
    return [
        {
            "comparator_id": str(row["comparator_id"]),
            "scenario_id": str(row["scenario_id"]),
            "object": object_name,
            "loss": float(row["loss"]),
        }
        for row in sorted(
            (row for row in rows if str(row["object"]) == object_name),
            key=lambda row: (str(row["comparator_id"]), str(row["scenario_id"])),
        )
    ]


def _build_dashboards(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    galaxies: Sequence[Any],
    packets: Sequence[Mapping[str, Any]],
    galaxy_results: Sequence[Mapping[str, Any]],
    cluster_results: Sequence[Mapping[str, Any]],
    galaxy_comparator_rows: Sequence[Mapping[str, Any]],
    cluster_comparator_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    item59_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    dashboards: dict[str, dict[str, Any]] = {}
    best_galaxy = min(
        galaxy_results, key=lambda row: (float(row["robust_loss"]), str(row["cell_id"]))
    )
    best_cluster = min(
        cluster_results, key=lambda row: (float(row["robust_loss"]), str(row["cell_id"]))
    )
    nominal_galaxy = next(
        row for row in _scenario_rows(config, "GALAXIES") if row["cell_id"] == "SPARC-ML-NOMINAL"
    )
    nominal_cluster = next(
        row for row in _scenario_rows(config, "CLUSTERS") if row["cell_id"] == "XCOP-SOURCE-NOMINAL"
    )
    for galaxy in galaxies:
        comparator = _normalized_dashboard_comparators(galaxy_comparator_rows, galaxy.name)
        top = _best_object_candidates(galaxy_results, galaxy.name)
        nominal = _nominal_sparc_prediction(
            str(best_galaxy["cell_id"]),
            manifest,
            context,
            galaxy,
            nominal_galaxy,
        )
        dashboards[f"GALAXIES-{galaxy.name}"] = {
            "domain": "GALAXIES",
            "object": galaxy.name,
            "status": _object_status(top, comparator),
            "top_candidates": top,
            "comparators": comparator,
            "unscored_required_comparators": [
                row
                for row in config["comparators"]
                if "BLOCKED" in str(row["status"]) or "XCOP" in str(row["status"])
            ],
            "overall_best_cell": best_galaxy["cell_id"],
            "overall_best_concept": best_galaxy["concept_id"],
            "source_profile": nominal["source_profile"],
            "candidate_state": nominal["candidate_state"],
            "nominal_radial_prediction": nominal["radial_prediction"],
            "environment": "SOURCE_BLOCKED_NO_ADMITTED_ENVIRONMENT_MAP",
        }
    for packet in packets:
        cluster = str(packet["cluster"])
        comparator = _normalized_dashboard_comparators(cluster_comparator_rows, cluster)
        top = _best_object_candidates(cluster_results, cluster)
        nominal = _nominal_xcop_prediction(
            str(best_cluster["cell_id"]),
            manifest,
            context,
            packet,
            nominal_cluster,
            item59_config,
        )
        dashboards[f"CLUSTERS-{cluster}"] = {
            "domain": "CLUSTERS",
            "object": cluster,
            "status": _object_status(top, comparator),
            "top_candidates": top,
            "comparators": comparator,
            "unscored_required_comparators": [
                row for row in config["comparators"] if "BLOCKED" in str(row["status"])
            ],
            "overall_best_cell": best_cluster["cell_id"],
            "overall_best_concept": best_cluster["concept_id"],
            "source_profile": nominal["source_profile"],
            "candidate_state": nominal["candidate_state"],
            "nominal_radial_prediction": nominal["radial_prediction"],
            "stellar_profile_status": (
                "AVAILABLE" if packet.get("stellar") is not None else "SHARED_GLOBAL_NUISANCE"
            ),
            "joint_pressure_temperature_covariance": "UNAVAILABLE",
            "shared_xray_measurement_ancestry": True,
            "morphology_sensitivity": "SOURCE_BLOCKED_NO_FROZEN_MORPHOLOGY_LEDGER",
            "spherical_approximation": "REQUIRED_AND_UNRESOLVED",
        }
    _require(len(dashboards) == 147, "dashboard count changed")
    return dashboards


def _closure_matrix(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in manifest["candidate_versions"]:
        rows.append(
            {
                "concept_id": candidate["candidate_id"],
                "lane": candidate["lane"],
                "candidate_status": candidate["candidate_status"],
                "domains": {
                    domain: candidate["domain_execution"][domain]["execution_disposition"]
                    for domain in DOMAINS
                },
                "physical_time_dilation_derived": False,
                "redshift_closure_derived": False,
                "capture_or_dissipation_derived": False,
                "light_propagation_derived": False,
                "tensor_or_quantum_gravity_derived": False,
            }
        )
    return rows


def _artifact_payloads(
    manifest: Mapping[str, Any],
    dashboards: Mapping[str, Mapping[str, Any]],
    galaxy_results: Sequence[Mapping[str, Any]],
    cluster_results: Sequence[Mapping[str, Any]],
    galaxy_adjudication: Sequence[Mapping[str, Any]],
    cluster_adjudication: Sequence[Mapping[str, Any]],
    cross_domain: Sequence[Mapping[str, Any]],
    comparator_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = {str(row["candidate_id"]): row for row in manifest["candidate_versions"]}
    counterexamples = [
        {
            "domain": row["domain"],
            "cell_id": row["cell_id"],
            "concept_id": row["concept_id"],
            "passes": row["passes"],
            "formula_sha256": candidates[str(row["concept_id"])]["formula_sha256"],
            "card_sha256": candidates[str(row["concept_id"])]["card_sha256"],
            "configuration_sha256": candidates[str(row["concept_id"])]["configuration_sha256"],
            "source_contract_sha256": candidates[str(row["concept_id"])]["domain_execution"][
                str(row["domain"])
            ]["source_contract_sha256"],
            "failure_class": "EMPIRICAL_DEVELOPMENT_COUNTEREXAMPLE",
            "failed_gates": sorted(key for key, value in row["gates"].items() if not value),
            "counterexample": row["counterexample"],
        }
        for row in [*galaxy_adjudication, *cluster_adjudication]
    ]
    blocked = [
        row
        for row in _closure_matrix(manifest)
        if row["candidate_status"] != "READY_FOR_RESPONSE_SCORING"
    ]
    survivors = [row for row in cross_domain if row["cross_domain_pass"]]
    artifacts: dict[str, Any] = {
        "global-cell-ledger.json": {
            "galaxies": galaxy_results,
            "clusters": cluster_results,
            "galaxy_adjudication": galaxy_adjudication,
            "cluster_adjudication": cluster_adjudication,
            "cross_domain": cross_domain,
            "dashboard_evidence": {
                key: {
                    "source_profile": dashboard["source_profile"],
                    "candidate_state": dashboard["candidate_state"],
                    "nominal_radial_prediction": dashboard["nominal_radial_prediction"],
                }
                for key, dashboard in sorted(dashboards.items())
            },
        },
        "counterexample-ledger.json": counterexamples,
        "closure-matrix.json": _closure_matrix(manifest),
        "blocked-idea-ledger.json": blocked,
        "comparator-ledger.json": comparator_summaries,
        "multiplicity-ledger.json": manifest["global_multiplicity_ledger"],
        "matched-environment-discriminator.json": {
            "status": "SOURCE_BLOCKED",
            "reason": "The admitted 139-galaxy ledger has no frozen isolated/satellite/filament/group environment map; group responses remain forbidden.",
            "negative_controls": {
                "environment_label_shuffle": "SOURCE_BLOCKED",
                "source_map_rotation_or_scramble": "SOURCE_BLOCKED",
                "equal_mass_randomized_neighbors": "SOURCE_BLOCKED",
                "L_g_zero": "COMPLETED_IN_TARGET_FREE_SYNTHETIC_GP01_FOUNDATION",
                "density_radius_external_field_only": "PARTLY_REPRESENTED_AS_SEPARATE_TWELL_DRIVERS",
                "catalog_completeness_and_distance_propagation": "SOURCE_BLOCKED",
                "leave_environment_class_out": "SOURCE_BLOCKED",
            },
        },
        "repair-ledger.json": {
            "post_freeze_repairs": [],
            "revision_budget": 0,
            "future_repairs_destination": registry.IDEA_RESERVOIR_ID,
        },
        "lay-summary.json": {
            "what_was_tested": (
                "Every source-ready frozen static radial parameter cell was tested separately "
                "on every admitted development galaxy or cluster, under three nuisance cases, "
                "with unchanged formulas between the pilot and full partitions."
            ),
            "what_was_not_tested": (
                "Temporal memory, field-line history, matched environment, group, lensing, "
                "redshift, capture, graviton, quantum-state, Solar, GW, and cosmology claims "
                "remain theory-only or source-blocked; the campaign did not turn a radial fit "
                "into evidence for those mechanisms."
            ),
            "cross_domain_survivor_count": len(survivors),
            "outcome": (
                "One or more exact shared cells passed every frozen development gate."
                if survivors
                else "No exact shared cell passed every frozen development gate; zero survivors is a valid result."
            ),
            "next_decisive_test": (
                "For any retained development signal, first derive the missing action and "
                "matter/light/GW closures. For spatial-history ideas, obtain an independently "
                "measured environment/source-history map before another response-scored campaign."
            ),
            "maximum_claim": "DEVELOPMENT_SIGNAL",
        },
    }
    for key, dashboard in dashboards.items():
        artifacts[f"dashboards/{key}.json"] = dashboard
    return artifacts


def _expected_artifact_paths(context: Mapping[str, Any]) -> tuple[str, ...]:
    dashboard_ids = [
        *(f"GALAXIES-{name}" for name in context["source_predecessor"]["objects"]["SPARC"]),
        *(f"CLUSTERS-{name}" for name in context["source_predecessor"]["objects"]["XCOP"]),
    ]
    relative = [
        *MANDATORY_CAMPAIGN_ARTIFACTS,
        *(f"dashboards/{dashboard_id}.json" for dashboard_id in dashboard_ids),
    ]
    _require(len(relative) == 156 and len(set(relative)) == 156, "artifact path contract changed")
    return tuple(sorted((ARTIFACT_DIRECTORY / path).as_posix() for path in relative))


def _require_finite_json(value: Any) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), "non-finite result value")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _require(isinstance(key, str), "non-string JSON key")
            _require_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _require_finite_json(child)


def _load_result_artifacts(
    root: Path,
    result: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    expected_paths = _expected_artifact_paths(context)
    index = result.get("artifact_index")
    _require(isinstance(index, list), "artifact index must be a list")
    _require(len(index) == 156, "artifact index count changed")
    indexed_paths: list[str] = []
    for row in index:
        _require(
            isinstance(row, Mapping) and set(row) == {"path", "sha256"},
            "artifact index row schema changed",
        )
        path = row.get("path")
        digest = row.get("sha256")
        _require(isinstance(path, str), "artifact path must be a string")
        _require(
            isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
            "artifact SHA-256 changed",
        )
        indexed_paths.append(path)
    _require(indexed_paths == list(expected_paths), "artifact index exact path order changed")
    _require(len(indexed_paths) == len(set(indexed_paths)), "duplicate artifact index path")

    artifact_root = (root / ARTIFACT_DIRECTORY).resolve()
    _require(artifact_root.is_dir(), "artifact directory missing")
    actual_paths = tuple(
        sorted(
            path.relative_to(root).as_posix() for path in artifact_root.rglob("*") if path.is_file()
        )
    )
    _require(actual_paths == expected_paths, "artifact directory exact file set changed")

    payloads: dict[str, Any] = {}
    prefix = f"{ARTIFACT_DIRECTORY.as_posix()}/"
    for row in index:
        indexed = str(row["path"])
        _require(indexed.startswith(prefix), "artifact path prefix changed")
        path = (root / indexed).resolve()
        _require(path.is_relative_to(artifact_root), "artifact path escaped")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == row["sha256"], f"artifact changed: {indexed}")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OpenGravityCampaignError("artifact is not canonical JSON") from error
        _require(canonical_bytes(payload) == raw, "artifact canonical JSON encoding changed")
        _require_finite_json(payload)
        payloads[indexed.removeprefix(prefix)] = payload
    return payloads


def _validate_score_rows(
    rows: Any,
    domain: str,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _require(isinstance(rows, list), f"{domain} score ledger must be a list")
    expected_cells = {
        str(cell["cell_id"]): str(candidate["candidate_id"])
        for candidate, cell in _eligible_cells(manifest, domain)
    }
    _require(len(rows) == len(expected_cells), f"{domain} score count changed")
    _require(
        {str(row.get("cell_id")) for row in rows if isinstance(row, Mapping)}
        == set(expected_cells),
        f"{domain} scored-cell set changed",
    )
    expected_scenarios = {str(row["cell_id"]) for row in _scenario_rows(config, domain)}
    expected_objects = set(
        map(
            str,
            context["source_predecessor"]["objects"]["SPARC" if domain == "GALAXIES" else "XCOP"],
        )
    )
    for row in rows:
        _require(isinstance(row, Mapping), f"{domain} score row schema changed")
        cell_id = str(row.get("cell_id"))
        _require(row.get("domain") == domain, f"{domain} score domain changed")
        _require(
            row.get("concept_id") == expected_cells[cell_id],
            f"{domain} score concept/cell link changed",
        )
        _require(
            isinstance(row.get("robust_loss"), (int, float))
            and not isinstance(row.get("robust_loss"), bool)
            and math.isfinite(float(row["robust_loss"]))
            and float(row["robust_loss"]) >= 0.0,
            f"{domain} robust loss changed",
        )
        scenarios = row.get("scenario_results")
        _require(isinstance(scenarios, list), f"{domain} scenario ledger changed")
        _require(
            {
                str(scenario.get("scenario_id"))
                for scenario in scenarios
                if isinstance(scenario, Mapping)
            }
            == expected_scenarios,
            f"{domain} nuisance scenario set changed",
        )
        for scenario in scenarios:
            _require(isinstance(scenario, Mapping), f"{domain} scenario row changed")
            objects = scenario.get("objects")
            _require(isinstance(objects, list), f"{domain} object score ledger changed")
            _require(
                {str(item.get("object")) for item in objects if isinstance(item, Mapping)}
                == expected_objects
                and len(objects) == len(expected_objects),
                f"{domain} object score set changed",
            )


def _validate_dashboard(
    dashboard_id: str,
    dashboard: Any,
    domain: str,
    object_name: str,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate_results: Sequence[Mapping[str, Any]],
    comparator_summary: Mapping[str, Any],
) -> int:
    _require(False, "legacy submitted-dashboard validator is forbidden")
    _require(isinstance(dashboard, Mapping), f"dashboard schema changed: {dashboard_id}")
    base_keys = {
        "domain",
        "object",
        "status",
        "top_candidates",
        "comparators",
        "unscored_required_comparators",
        "overall_best_cell",
        "overall_best_concept",
        "source_profile",
        "candidate_state",
        "nominal_radial_prediction",
    }
    extra_keys = (
        {"environment"}
        if domain == "GALAXIES"
        else {
            "stellar_profile_status",
            "joint_pressure_temperature_covariance",
            "shared_xray_measurement_ancestry",
            "morphology_sensitivity",
            "spherical_approximation",
        }
    )
    _require(set(dashboard) == base_keys | extra_keys, f"dashboard keys changed: {dashboard_id}")
    _require(
        dashboard.get("domain") == domain and dashboard.get("object") == object_name,
        f"dashboard object link changed: {dashboard_id}",
    )
    _require(
        dashboard.get("status") in {"SUPPORTS", "COUNTEREXAMPLE", "NEUTRAL", "SOURCE_BLOCKED"},
        f"dashboard status changed: {dashboard_id}",
    )
    best_cell = str(result["best_development_cells"][domain])
    cells = {str(row["cell_id"]): row for row in manifest["parameter_cells"]}
    _require(
        dashboard.get("overall_best_cell") == best_cell,
        f"dashboard best cell changed: {dashboard_id}",
    )
    _require(
        dashboard.get("overall_best_concept")
        == cells[best_cell]["exact_value_or_rule"]["concept_id"],
        f"dashboard best concept changed: {dashboard_id}",
    )
    _require(
        dashboard.get("top_candidates") == _best_object_candidates(candidate_results, object_name),
        f"dashboard candidate ranking changed: {dashboard_id}",
    )
    expected_comparators = sorted(
        (
            str(scenario["comparator_id"]),
            str(scenario["scenario_id"]),
            str(item["object"]),
            float(item["loss"]),
        )
        for scenario in comparator_summary["scenario_results"]
        for item in scenario["objects"]
        if str(item["object"]) == object_name
    )
    actual_comparators = sorted(
        (
            str(row["comparator_id"]),
            str(row["scenario_id"]),
            str(row["object"]),
            float(row["loss"]),
        )
        for row in dashboard["comparators"]
    )
    _require(
        actual_comparators == expected_comparators,
        f"dashboard comparator link changed: {dashboard_id}",
    )
    source_profile = dashboard.get("source_profile")
    candidate_state = dashboard.get("candidate_state")
    predictions = dashboard.get("nominal_radial_prediction")
    _require(
        isinstance(source_profile, list)
        and isinstance(candidate_state, list)
        and isinstance(predictions, list)
        and source_profile
        and len(source_profile) == len(candidate_state)
        and predictions,
        f"dashboard radial evidence changed: {dashboard_id}",
    )
    if domain == "GALAXIES":
        _require(
            len(predictions) == len(source_profile),
            f"galaxy dashboard row count changed: {dashboard_id}",
        )
    for row in predictions:
        _require(
            isinstance(row, Mapping)
            and isinstance(row.get("observed"), (int, float))
            and not isinstance(row.get("observed"), bool)
            and isinstance(row.get("error"), (int, float))
            and not isinstance(row.get("error"), bool)
            and float(row["error"]) > 0.0,
            f"dashboard prediction schema changed: {dashboard_id}",
        )
    return len(predictions)


def _dashboard_comparators_from_summary(
    comparator_summary: Mapping[str, Any], object_name: str
) -> list[dict[str, Any]]:
    return [
        {
            "comparator_id": str(scenario["comparator_id"]),
            "scenario_id": str(scenario["scenario_id"]),
            "object": object_name,
            "loss": float(item["loss"]),
        }
        for scenario in comparator_summary["scenario_results"]
        for item in scenario["objects"]
        if str(item["object"]) == object_name
    ]


def _rebuild_dashboard(
    evidence: Mapping[str, Any],
    domain: str,
    object_name: str,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate_results: Sequence[Mapping[str, Any]],
    comparator_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    best_cell = str(result["best_development_cells"][domain])
    cells = {str(row["cell_id"]): row for row in manifest["parameter_cells"]}
    comparators = _dashboard_comparators_from_summary(comparator_summary, object_name)
    top = _best_object_candidates(candidate_results, object_name)
    dashboard: dict[str, Any] = {
        "domain": domain,
        "object": object_name,
        "status": _object_status(top, comparators),
        "top_candidates": top,
        "comparators": comparators,
        "unscored_required_comparators": [
            row
            for row in config["comparators"]
            if "BLOCKED" in str(row["status"])
            or (domain == "GALAXIES" and "XCOP" in str(row["status"]))
        ],
        "overall_best_cell": best_cell,
        "overall_best_concept": cells[best_cell]["exact_value_or_rule"]["concept_id"],
        "source_profile": evidence["source_profile"],
        "candidate_state": evidence["candidate_state"],
        "nominal_radial_prediction": evidence["nominal_radial_prediction"],
    }
    if domain == "GALAXIES":
        dashboard["environment"] = "SOURCE_BLOCKED_NO_ADMITTED_ENVIRONMENT_MAP"
    else:
        dashboard.update(
            {
                "stellar_profile_status": (
                    "AVAILABLE"
                    if object_name in config["data_contract"]["xcop_stellar_available"]
                    else "SHARED_GLOBAL_NUISANCE"
                ),
                "joint_pressure_temperature_covariance": "UNAVAILABLE",
                "shared_xray_measurement_ancestry": True,
                "morphology_sensitivity": "SOURCE_BLOCKED_NO_FROZEN_MORPHOLOGY_LEDGER",
                "spherical_approximation": "REQUIRED_AND_UNRESOLVED",
            }
        )
    return dashboard


def _nominal_object_score(
    candidate_results: Sequence[Mapping[str, Any]],
    best_cell: str,
    scenario_id: str,
    object_name: str,
) -> Mapping[str, Any]:
    candidate = next(row for row in candidate_results if str(row["cell_id"]) == best_cell)
    scenario = next(
        row for row in candidate["scenario_results"] if str(row["scenario_id"]) == scenario_id
    )
    return next(row for row in scenario["objects"] if str(row["object"]) == object_name)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def _validate_dashboard_evidence(
    dashboard_id: str,
    evidence: Any,
    domain: str,
    object_name: str,
    result: Mapping[str, Any],
    candidate_results: Sequence[Mapping[str, Any]],
) -> int:
    _require(
        isinstance(evidence, Mapping)
        and set(evidence) == {"source_profile", "candidate_state", "nominal_radial_prediction"},
        f"dashboard evidence keys changed: {dashboard_id}",
    )
    source_profile = evidence["source_profile"]
    candidate_state = evidence["candidate_state"]
    predictions = evidence["nominal_radial_prediction"]
    _require(
        isinstance(source_profile, list)
        and isinstance(candidate_state, list)
        and isinstance(predictions, list)
        and source_profile
        and len(source_profile) == len(candidate_state)
        and predictions,
        f"dashboard radial evidence changed: {dashboard_id}",
    )
    source_keys = (
        {"radius_kpc", "baryonic_acceleration_m_s2"}
        if domain == "GALAXIES"
        else {
            "radius_kpc",
            "baryonic_acceleration_m_s2",
            "gas_density_kg_m3",
            "baryonic_enclosed_mass_kg",
        }
    )
    for source, state in zip(source_profile, candidate_state, strict=True):
        _require(
            isinstance(source, Mapping)
            and set(source) == source_keys
            and isinstance(state, Mapping)
            and set(state) == {"radius_kpc", "gain_factor"},
            f"dashboard source/state schema changed: {dashboard_id}",
        )
        radius = float(source["radius_kpc"])
        _require(
            radius >= 0.0
            and float(state["radius_kpc"]) == radius
            and float(source["baryonic_acceleration_m_s2"]) > 0.0
            and float(state["gain_factor"]) > 0.0,
            f"dashboard source/state values changed: {dashboard_id}",
        )
        if domain == "CLUSTERS":
            _require(
                float(source["gas_density_kg_m3"]) > 0.0
                and float(source["baryonic_enclosed_mass_kg"]) >= 0.0,
                f"cluster dashboard source values changed: {dashboard_id}",
            )
            if radius > 0.0:
                expected_acceleration = (
                    6.6743e-11
                    * float(source["baryonic_enclosed_mass_kg"])
                    / (radius * 3.085677581491367e19) ** 2
                )
                _require(
                    _close(float(source["baryonic_acceleration_m_s2"]), expected_acceleration),
                    f"cluster dashboard mass/acceleration link changed: {dashboard_id}",
                )

    best_cell = str(result["best_development_cells"][domain])
    if domain == "GALAXIES":
        _require(
            len(predictions) == len(source_profile),
            f"galaxy dashboard row count changed: {dashboard_id}",
        )
        squares = []
        for source, state, row in zip(source_profile, candidate_state, predictions, strict=True):
            _require(
                isinstance(row, Mapping)
                and set(row) == {"radius_kpc", "observed", "error", "predicted"},
                f"galaxy dashboard prediction schema changed: {dashboard_id}",
            )
            radius = float(source["radius_kpc"])
            observed = float(row["observed"])
            error = float(row["error"])
            predicted = float(row["predicted"])
            expected_prediction = math.sqrt(
                float(state["gain_factor"])
                * float(source["baryonic_acceleration_m_s2"])
                * radius
                * 3.085677581491367e19
                / 1.0e6
            )
            _require(
                float(row["radius_kpc"]) == radius
                and observed > 0.0
                and error > 0.0
                and predicted > 0.0
                and _close(predicted, expected_prediction),
                f"galaxy dashboard prediction/source link changed: {dashboard_id}",
            )
            squares.append(((predicted - observed) / error) ** 2)
        expected_score = _nominal_object_score(
            candidate_results, best_cell, "SPARC-ML-NOMINAL", object_name
        )
        _require(
            int(expected_score["row_count"]) == len(predictions)
            and _close(float(expected_score["loss"]), float(np.mean(squares)))
            and _close(float(expected_score["worst_standardized_square"]), max(squares)),
            f"galaxy dashboard score link changed: {dashboard_id}",
        )
    else:
        rows: list[dict[str, Any]] = []
        predicted_by_id: dict[str, float] = {}
        for row in predictions:
            _require(
                isinstance(row, Mapping)
                and set(row)
                == {"row_id", "observable", "radius_kpc", "observed", "error", "predicted"},
                f"cluster dashboard prediction schema changed: {dashboard_id}",
            )
            row_id = str(row["row_id"])
            _require(
                row_id not in predicted_by_id,
                f"duplicate cluster response row: {dashboard_id}",
            )
            _require(
                row["observable"] in {"pressure", "temperature"}
                and float(row["radius_kpc"]) > 0.0
                and float(row["observed"]) > 0.0
                and float(row["error"]) >= 0.0
                and float(row["predicted"]) > 0.0,
                f"cluster dashboard prediction values changed: {dashboard_id}",
            )
            rows.append(dict(row))
            predicted_by_id[row_id] = float(row["predicted"])
        score = _loss_rows(predicted_by_id, rows, minimum_fractional_error=0.05)
        expected_score = _nominal_object_score(
            candidate_results, best_cell, "XCOP-SOURCE-NOMINAL", object_name
        )
        _require(
            int(expected_score["row_count"]) == int(score["row_count"])
            and _close(float(expected_score["loss"]), float(score["loss"]))
            and expected_score["by_observable"] == score["by_observable"]
            and _close(
                float(expected_score["worst_standardized_square"]),
                float(score["worst_standardized_square"]),
            ),
            f"cluster dashboard score link changed: {dashboard_id}",
        )
    return len(predictions)


def _validate_result_artifacts(
    result: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    expected_relative = {
        path.removeprefix(f"{ARTIFACT_DIRECTORY.as_posix()}/")
        for path in _expected_artifact_paths(context)
    }
    _require(set(artifacts) == expected_relative, "result artifact payload set changed")
    global_ledger = artifacts.get("global-cell-ledger.json")
    comparator_ledger = artifacts.get("comparator-ledger.json")
    _require(
        isinstance(global_ledger, Mapping)
        and set(global_ledger)
        == {
            "galaxies",
            "clusters",
            "galaxy_adjudication",
            "cluster_adjudication",
            "cross_domain",
            "dashboard_evidence",
        },
        "global cell ledger schema changed",
    )
    _require(
        isinstance(comparator_ledger, Mapping)
        and set(comparator_ledger) == {"GALAXIES", "CLUSTERS", "declared_source_or_solver_blocked"},
        "comparator ledger schema changed",
    )
    galaxies = global_ledger["galaxies"]
    clusters = global_ledger["clusters"]
    _validate_score_rows(galaxies, "GALAXIES", manifest, context, config)
    _validate_score_rows(clusters, "CLUSTERS", manifest, context, config)
    _require(
        comparator_ledger["GALAXIES"].get("domain") == "GALAXIES",
        "galaxy comparator domain changed",
    )
    _require(
        comparator_ledger["CLUSTERS"].get("domain") == "CLUSTERS",
        "cluster comparator domain changed",
    )
    expected_galaxy_adjudication = _adjudicate_domain(
        "GALAXIES", galaxies, comparator_ledger["GALAXIES"], context, config
    )
    expected_cluster_adjudication = _adjudicate_domain(
        "CLUSTERS", clusters, comparator_ledger["CLUSTERS"], context, config
    )
    _require(
        global_ledger["galaxy_adjudication"] == expected_galaxy_adjudication,
        "galaxy adjudication does not rebuild",
    )
    _require(
        global_ledger["cluster_adjudication"] == expected_cluster_adjudication,
        "cluster adjudication does not rebuild",
    )
    expected_cross_domain = _cross_domain_adjudication(
        expected_galaxy_adjudication, expected_cluster_adjudication
    )
    _require(
        global_ledger["cross_domain"] == expected_cross_domain,
        "cross-domain adjudication does not rebuild",
    )
    _require(
        result.get("cross_domain_adjudication") == expected_cross_domain,
        "result cross-domain ledger changed",
    )
    _require(
        result.get("cross_domain_survivors")
        == [row for row in expected_cross_domain if row["cross_domain_pass"]],
        "result survivor ledger changed",
    )
    _require(
        result.get("best_development_cells")
        == {
            "GALAXIES": min(
                galaxies, key=lambda row: (float(row["robust_loss"]), str(row["cell_id"]))
            )["cell_id"],
            "CLUSTERS": min(
                clusters, key=lambda row: (float(row["robust_loss"]), str(row["cell_id"]))
            )["cell_id"],
        },
        "best development cell changed",
    )

    dashboard_payloads: dict[str, Mapping[str, Any]] = {}
    dashboard_evidence = global_ledger["dashboard_evidence"]
    _require(
        isinstance(dashboard_evidence, Mapping)
        and set(dashboard_evidence)
        == {
            path.removeprefix("dashboards/").removesuffix(".json")
            for path in expected_relative
            if path.startswith("dashboards/")
        },
        "dashboard evidence object set changed",
    )
    sparc_rows = 0
    xcop_rows = 0
    for domain, names, candidate_rows, comparator_summary in (
        (
            "GALAXIES",
            context["source_predecessor"]["objects"]["SPARC"],
            galaxies,
            comparator_ledger["GALAXIES"],
        ),
        (
            "CLUSTERS",
            context["source_predecessor"]["objects"]["XCOP"],
            clusters,
            comparator_ledger["CLUSTERS"],
        ),
    ):
        for object_name in names:
            dashboard_id = f"{domain}-{object_name}"
            evidence = dashboard_evidence[dashboard_id]
            row_count = _validate_dashboard_evidence(
                dashboard_id,
                evidence,
                domain,
                str(object_name),
                result,
                candidate_rows,
            )
            rebuilt_dashboard = _rebuild_dashboard(
                evidence,
                domain,
                str(object_name),
                result,
                manifest,
                candidate_rows,
                comparator_summary,
                config,
            )
            _require(
                artifacts[f"dashboards/{dashboard_id}.json"] == rebuilt_dashboard,
                f"dashboard does not rebuild: {dashboard_id}",
            )
            dashboard_payloads[dashboard_id] = rebuilt_dashboard
            if domain == "GALAXIES":
                sparc_rows += row_count
            else:
                xcop_rows += row_count
    _require(sparc_rows == 2720, "SPARC dashboard response-row total changed")
    _require(xcop_rows == 184, "X-COP dashboard response-row total changed")

    expected_counts = {
        "live_candidates": 407,
        "parameter_cells": 2486,
        "galaxy_cells_scored": 179,
        "cluster_cells_scored": 1669,
        "galaxies_scored": 139,
        "clusters_scored": 8,
        "sparc_rows_parsed": 3391,
        "sparc_rows_scored": 2720,
        "xcop_response_rows_scored": 184,
        "scientific_response_unique_files_opened": 17,
        "scientific_source_unique_files_opened": 13,
        "unique_local_payload_files_opened": 30,
        "committed_sparc_blob_verification_reads": 1,
        "local_payload_read_operations": 31,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "tuning_calls": 0,
        "dashboards": 147,
        "artifacts": 156,
    }
    _require(result.get("counts") == expected_counts, "result exact counts changed")

    expected_payloads = _artifact_payloads(
        manifest,
        dashboard_payloads,
        galaxies,
        clusters,
        expected_galaxy_adjudication,
        expected_cluster_adjudication,
        expected_cross_domain,
        comparator_ledger,
    )
    _require(dict(artifacts) == expected_payloads, "artifact schemas or cross-links do not rebuild")


def _result_adjudication(
    result: Mapping[str, Any], artifact_index: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    adjudication: dict[str, Any] = {
        "schema_version": ADJUDICATION_SCHEMA,
        "campaign_id": result["campaign_id"],
        "result_content_sha256": result["result_content_sha256"],
        "artifact_index_sha256": content_sha256(artifact_index),
        "validated_counts": result["counts"],
        "survivor_count": len(result["cross_domain_survivors"]),
        "maximum_label": "DEVELOPMENT_SIGNAL",
        "session_terminal": True,
        "second_campaign_allowed": False,
        "adjudication_content_sha256": "",
    }
    adjudication["adjudication_content_sha256"] = _self_hash(
        adjudication, "adjudication_content_sha256"
    )
    return adjudication


def _recompute_campaign_outputs_from_frozen_inputs(
    root: Path,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently rebuild every scientific result artifact from frozen local inputs."""

    clock_config = _verify_scientific_input_contracts(root, config)
    galaxies, sparc_provenance = _load_sparc_responses(root, context, config)
    packets, _xcop_files, clock_config, item59_config = _load_xcop_responses(
        root, config, clock_config
    )
    galaxy_scenarios = _scenario_rows(config, "GALAXIES")
    cluster_scenarios = _scenario_rows(config, "CLUSTERS")
    galaxy_comparator_rows, _galaxy_comparator_map = _score_sparc_comparators(
        galaxies, galaxy_scenarios
    )
    cluster_comparator_rows, _cluster_comparator_map = _score_xcop_comparators(
        packets,
        cluster_scenarios,
        clock_config,
        item59_config,
    )
    galaxy_results = _score_sparc_candidates(manifest, context, galaxies, galaxy_scenarios)
    cluster_results = _score_xcop_candidates(
        manifest, context, packets, cluster_scenarios, item59_config
    )
    galaxy_comparators = _comparator_summary(galaxy_comparator_rows, "GALAXIES")
    cluster_comparators = _comparator_summary(cluster_comparator_rows, "CLUSTERS")
    galaxy_adjudication = _adjudicate_domain(
        "GALAXIES", galaxy_results, galaxy_comparators, context, config
    )
    cluster_adjudication = _adjudicate_domain(
        "CLUSTERS", cluster_results, cluster_comparators, context, config
    )
    cross_domain = _cross_domain_adjudication(galaxy_adjudication, cluster_adjudication)
    dashboards = _build_dashboards(
        manifest,
        context,
        galaxies,
        packets,
        galaxy_results,
        cluster_results,
        galaxy_comparator_rows,
        cluster_comparator_rows,
        config,
        item59_config,
    )
    comparator_summaries = {
        "GALAXIES": galaxy_comparators,
        "CLUSTERS": cluster_comparators,
        "declared_source_or_solver_blocked": [
            row for row in config["comparators"] if "BLOCKED" in str(row["status"])
        ],
    }
    artifact_payloads = _artifact_payloads(
        manifest,
        dashboards,
        galaxy_results,
        cluster_results,
        galaxy_adjudication,
        cluster_adjudication,
        cross_domain,
        comparator_summaries,
    )
    _require(len(artifact_payloads) == 156, "recomputed artifact count changed")
    xcop_rows = sum(len(packet["rows"]) for packet in packets)
    _require(xcop_rows == 184, "recomputed X-COP response-row count changed")
    counts = {
        "live_candidates": 407,
        "parameter_cells": 2486,
        "galaxy_cells_scored": len(galaxy_results),
        "cluster_cells_scored": len(cluster_results),
        "galaxies_scored": len(galaxies),
        "clusters_scored": len(packets),
        "sparc_rows_parsed": int(sparc_provenance["point_count"]),
        "sparc_rows_scored": sum(galaxy.count for galaxy in galaxies),
        "xcop_response_rows_scored": xcop_rows,
        "scientific_response_unique_files_opened": 17,
        "scientific_source_unique_files_opened": 13,
        "unique_local_payload_files_opened": 30,
        "committed_sparc_blob_verification_reads": 1,
        "local_payload_read_operations": 31,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "tuning_calls": 0,
        "dashboards": len(dashboards),
        "artifacts": len(artifact_payloads),
    }
    return {
        "artifact_payloads": _jsonable(artifact_payloads),
        "counts": counts,
        "cross_domain_adjudication": _jsonable(cross_domain),
        "cross_domain_survivors": _jsonable(
            [row for row in cross_domain if row["cross_domain_pass"]]
        ),
        "best_development_cells": {
            "GALAXIES": min(
                galaxy_results,
                key=lambda row: (float(row["robust_loss"]), str(row["cell_id"])),
            )["cell_id"],
            "CLUSTERS": min(
                cluster_results,
                key=lambda row: (float(row["robust_loss"]), str(row["cell_id"])),
            )["cell_id"],
        },
    }


def execute_campaign() -> dict[str, Any]:
    root = _repo_root()
    for path in (
        ACCESS_INTENT_PATH,
        RESULT_PATH,
        ADJUDICATION_PATH,
        FAILURE_PATH,
    ):
        _require(not (root / path).exists(), f"campaign replay refused: {path}")
    _require(not (root / ARTIFACT_DIRECTORY).exists(), "campaign artifact directory already exists")
    package_commit = _require_committed_preflight(root)
    manifest, context = build_manifest(root)
    intent = build_access_intent(root, package_commit)
    intent_status = _atomic_no_clobber(root / ACCESS_INTENT_PATH, canonical_bytes(intent))
    _require(intent_status == "CREATED", "access-intent replay/refusal")
    try:
        config = load_config(root)
        clock_config = _verify_scientific_input_contracts(root, config)
        galaxies, sparc_provenance = _load_sparc_responses(root, context, config)
        packets, _xcop_files, clock_config, item59_config = _load_xcop_responses(
            root, config, clock_config
        )
        galaxy_scenarios = _scenario_rows(config, "GALAXIES")
        cluster_scenarios = _scenario_rows(config, "CLUSTERS")
        galaxy_comparator_rows, _galaxy_comparator_map = _score_sparc_comparators(
            galaxies, galaxy_scenarios
        )
        cluster_comparator_rows, _cluster_comparator_map = _score_xcop_comparators(
            packets,
            cluster_scenarios,
            clock_config,
            item59_config,
        )
        galaxy_results = _score_sparc_candidates(manifest, context, galaxies, galaxy_scenarios)
        cluster_results = _score_xcop_candidates(
            manifest, context, packets, cluster_scenarios, item59_config
        )
        galaxy_comparators = _comparator_summary(galaxy_comparator_rows, "GALAXIES")
        cluster_comparators = _comparator_summary(cluster_comparator_rows, "CLUSTERS")
        galaxy_adjudication = _adjudicate_domain(
            "GALAXIES", galaxy_results, galaxy_comparators, context, config
        )
        cluster_adjudication = _adjudicate_domain(
            "CLUSTERS", cluster_results, cluster_comparators, context, config
        )
        cross_domain = _cross_domain_adjudication(galaxy_adjudication, cluster_adjudication)
        survivors = [row for row in cross_domain if row["cross_domain_pass"]]
        dashboards = _build_dashboards(
            manifest,
            context,
            galaxies,
            packets,
            galaxy_results,
            cluster_results,
            galaxy_comparator_rows,
            cluster_comparator_rows,
            config,
            item59_config,
        )
        comparator_summaries = {
            "GALAXIES": galaxy_comparators,
            "CLUSTERS": cluster_comparators,
            "declared_source_or_solver_blocked": [
                row for row in config["comparators"] if "BLOCKED" in str(row["status"])
            ],
        }
        artifact_payloads = _artifact_payloads(
            manifest,
            dashboards,
            galaxy_results,
            cluster_results,
            galaxy_adjudication,
            cluster_adjudication,
            cross_domain,
            comparator_summaries,
        )
        _require(len(artifact_payloads) == 156, "result artifact count changed")
        _require(
            sum(len(packet["rows"]) for packet in packets) == 184,
            "X-COP response-row count changed",
        )
        artifact_index = [
            {
                "path": (ARTIFACT_DIRECTORY / relative).as_posix(),
                "sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
            }
            for relative, payload in sorted(artifact_payloads.items())
        ]
        counts = {
            "live_candidates": 407,
            "parameter_cells": 2486,
            "galaxy_cells_scored": len(galaxy_results),
            "cluster_cells_scored": len(cluster_results),
            "galaxies_scored": len(galaxies),
            "clusters_scored": len(packets),
            "sparc_rows_parsed": int(sparc_provenance["point_count"]),
            "sparc_rows_scored": sum(galaxy.count for galaxy in galaxies),
            "xcop_response_rows_scored": sum(len(packet["rows"]) for packet in packets),
            "scientific_response_unique_files_opened": 17,
            "scientific_source_unique_files_opened": 13,
            "unique_local_payload_files_opened": 30,
            "committed_sparc_blob_verification_reads": 1,
            "local_payload_read_operations": 31,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "tuning_calls": 0,
            "dashboards": len(dashboards),
            "artifacts": len(artifact_payloads),
        }
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "campaign_id": manifest["campaign_id"],
            "status": "DEVELOPMENT_CAMPAIGN_COMPLETE_SESSION_TERMINAL",
            "package_commit": package_commit,
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "terminal_ledger_content_sha256": _read_json(root / TERMINAL_LEDGER_PATH)[
                "ledger_content_sha256"
            ],
            "access_intent_content_sha256": intent["intent_content_sha256"],
            "counts": counts,
            "cross_domain_survivors": survivors,
            "cross_domain_adjudication": cross_domain,
            "best_development_cells": {
                "GALAXIES": min(
                    galaxy_results,
                    key=lambda row: (float(row["robust_loss"]), str(row["cell_id"])),
                )["cell_id"],
                "CLUSTERS": min(
                    cluster_results,
                    key=lambda row: (float(row["robust_loss"]), str(row["cell_id"])),
                )["cell_id"],
            },
            "artifact_index": artifact_index,
            "claim_ceiling": config["claim_ceiling"],
            "maximum_label": "DEVELOPMENT_SIGNAL",
            "global_discovery_p_value": None,
            "external_cost_usd": 0.0,
            "result_content_sha256": "",
        }
        result = _jsonable(result)
        result["result_content_sha256"] = _self_hash(result, "result_content_sha256")
        for relative, payload in sorted(artifact_payloads.items()):
            status = _atomic_no_clobber(
                root / ARTIFACT_DIRECTORY / relative, canonical_bytes(_jsonable(payload))
            )
            _require(status == "CREATED", f"artifact publication failed: {relative}")
        _require(
            _atomic_no_clobber(root / RESULT_PATH, canonical_bytes(result)) == "CREATED",
            "result publication failed",
        )
        adjudication = _result_adjudication(result, artifact_index)
        _require(
            _atomic_no_clobber(root / ADJUDICATION_PATH, canonical_bytes(adjudication))
            == "CREATED",
            "adjudication publication failed",
        )
        return result
    except BaseException as error:
        error_code = "INTERNAL_FAILURE"
        if isinstance(error, OpenGravityCampaignError):
            error_code = "CONTRACT_OR_SCIENTIFIC_GATE_FAILURE"
        elif isinstance(error, OSError):
            error_code = "LOCAL_PAYLOAD_OR_PUBLICATION_IO_FAILURE"
        elif isinstance(error, MemoryError):
            error_code = "MEMORY_FAILURE"
        elif isinstance(error, KeyboardInterrupt):
            error_code = "INTERRUPTED"
        failure: dict[str, Any] = {
            "schema_version": FAILURE_SCHEMA,
            "campaign_id": "OPEN-GRAVITY-CAMPAIGN-v1",
            "status": "TERMINAL_FAILURE_SUCCESSOR_REQUIRED",
            "error_code": error_code,
            "raw_exception_message_retained": False,
            "raw_exception_class_retained": False,
            "access_intent_exists": (root / ACCESS_INTENT_PATH).exists(),
            "result_exists": (root / RESULT_PATH).exists(),
            "adjudication_exists": (root / ADJUDICATION_PATH).exists(),
            "replay_allowed": False,
            "failure_content_sha256": "",
        }
        failure["failure_content_sha256"] = _self_hash(failure, "failure_content_sha256")
        _atomic_no_clobber(root / FAILURE_PATH, canonical_bytes(failure))
        raise


def check_result() -> dict[str, Any]:
    root = _repo_root()
    _require(not (root / FAILURE_PATH).exists(), "terminal failure receipt exists")
    package_commit = _require_committed_preflight(root)
    manifest, context = build_manifest(root)
    config = load_config(root)
    intent = _read_json(root / ACCESS_INTENT_PATH)
    result = _read_json(root / RESULT_PATH)
    adjudication = _read_json(root / ADJUDICATION_PATH)
    _require(intent == build_access_intent(root, package_commit), "access intent changed")
    _require(
        set(result)
        == {
            "schema_version",
            "campaign_id",
            "status",
            "package_commit",
            "manifest_content_sha256",
            "terminal_ledger_content_sha256",
            "access_intent_content_sha256",
            "counts",
            "cross_domain_survivors",
            "cross_domain_adjudication",
            "best_development_cells",
            "artifact_index",
            "claim_ceiling",
            "maximum_label",
            "global_discovery_p_value",
            "external_cost_usd",
            "result_content_sha256",
        },
        "result key set changed",
    )
    _require(result.get("schema_version") == RESULT_SCHEMA, "result schema changed")
    _require(result.get("campaign_id") == manifest["campaign_id"], "result campaign changed")
    _require(
        result.get("status") == "DEVELOPMENT_CAMPAIGN_COMPLETE_SESSION_TERMINAL",
        "result completion status changed",
    )
    _require(result.get("package_commit") == package_commit, "result package changed")
    _require(
        result.get("manifest_content_sha256") == manifest["manifest_content_sha256"],
        "result manifest changed",
    )
    _require(
        result.get("result_content_sha256") == _self_hash(result, "result_content_sha256"),
        "result self hash changed",
    )
    _require(
        result.get("terminal_ledger_content_sha256")
        == build_terminal_ledger(manifest)["ledger_content_sha256"],
        "result terminal ledger changed",
    )
    _require(
        result.get("access_intent_content_sha256") == intent["intent_content_sha256"],
        "result access intent changed",
    )
    _require(result.get("claim_ceiling") == config["claim_ceiling"], "result claim ceiling changed")
    _require(result.get("global_discovery_p_value") is None, "global discovery p-value overclaim")
    _require(float(result.get("external_cost_usd", -1.0)) == 0.0, "result external cost changed")
    _require_finite_json(result)
    artifacts = _load_result_artifacts(root, result, context)
    try:
        _validate_result_artifacts(result, artifacts, manifest, context, config)
    except OpenGravityCampaignError:
        raise
    except (KeyError, TypeError, ValueError, StopIteration) as error:
        raise OpenGravityCampaignError("result artifact schema or cross-link invalid") from error
    recomputed = _recompute_campaign_outputs_from_frozen_inputs(root, manifest, context, config)
    _require(
        artifacts == recomputed["artifact_payloads"],
        "stored artifacts differ from independent frozen-input recomputation",
    )
    for field in (
        "counts",
        "cross_domain_adjudication",
        "cross_domain_survivors",
        "best_development_cells",
    ):
        _require(
            result[field] == recomputed[field],
            f"result {field} differs from independent frozen-input recomputation",
        )
    expected_adjudication = _result_adjudication(result, result["artifact_index"])
    _require(adjudication == expected_adjudication, "adjudication changed")
    _require(result["maximum_label"] == "DEVELOPMENT_SIGNAL", "result overclaims")
    return adjudication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "check-preflight", "status", "execute", "check-result")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(write_preflight(), sort_keys=True))
    elif args.command == "check-preflight":
        receipt = check_preflight()
        print(
            json.dumps(
                {
                    "decision": receipt["decision"],
                    "candidate_count": receipt["manifest"]["candidate_count"],
                    "parameter_cell_count": receipt["manifest"]["parameter_cell_count"],
                    "planned_domain_executions": receipt["manifest"]["planned_domain_executions"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "status":
        print(
            json.dumps(
                {
                    "manifest_exists": MANIFEST_PATH.exists(),
                    "terminal_ledger_exists": TERMINAL_LEDGER_PATH.exists(),
                    "access_intent_exists": ACCESS_INTENT_PATH.exists(),
                    "result_exists": RESULT_PATH.exists(),
                    "failure_exists": FAILURE_PATH.exists(),
                },
                sort_keys=True,
            )
        )
    elif args.command == "execute":
        execute_campaign()
    else:
        check_result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
