"""Final registry-bound TWELL-400 exact-cell theory packet compiler.

The packet is a complete live-card and campaign-manifest input, not a campaign.  Its
domain dispositions remain theory-only until a separately audited source matrix is
bound.  No response-bearing payload is opened by this compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sigma_theory_compiler import open_gravity_registry_foundation_v1 as registry
from sigma_theory_compiler import twell_400_v2_typed_compiler_packet as interim

CONFIG_PATH = Path("configs/twell_400_v2_typed_compiler_final_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/twell_400_v2_typed_compiler_final_v2.py")
TEST_PATH = Path("tests/test_twell_400_v2_typed_compiler_final_v2.py")
CARDS_PATH = Path("runs/gravity/twell-400-v2-typed-compiler-final-v2/cards.jsonl")
RECEIPT_PATH = Path("runs/gravity/twell-400-v2-typed-compiler-final-v2/receipt.json")
PACKET_ID = "TWELL-400-v2-TYPED-COMPILER-FINAL-v2"
DECISION = (
    "PASS_FINAL_HARD_BOUND_TWELL_EXACT_CELL_MANIFEST_INPUT_SOURCE_MATRIX_DEFERRED_"
    "NO_CAMPAIGN_AUTHORITY"
)
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "9e13e0d5fb4d409dc2f8edbcac0688091b7d991fd347bcffae9f7ac95a603578"
)
EXPECTED_UNSEALED_ROOT_SHA256 = "e1dc46c710c2f3f51dc045b19391709693ad354dd8708afb9f9aa94d295ecce4"
EXPECTED_SECTION_SEALS = {
    "identity": "744ecf09f855c3a05a5edbd7200987d6e7f6e07496b982713d391a9efb611b5b",
    "hard_bindings": "247a165c02b70bd4e06d412e3209a6709087263b9c57434f0becd1184014f112",
    "deferred_bindings": "5ed8530102389bc3622ef98da95ff1774f39be656e816654d9ce043073acc7e9",
    "commit_provenance": "b6206f0a5728c923a50c572630d45bbf180ecf082b6b7b2f63b8bd099dc62163",
    "compiler_contract": "1f77283c3b2327b42ce4899a8f6849d877ca0cec09c150ee144991f0bed4e738",
    "lane_contract": "cca853c97d6cf1a9b15b78944d4e6133dfa363bfced38ef65cc7740d3a10ad35",
    "source_gate_contract": "3f519d7077e64e9deb328939860668b593f0f47a516bd5bfa1c20b8318f3ad2a",
    "access_contract": "7065dc9638caeb8b37395549010d3842c64ec201c658a1e624cefcd3a65c9079",
    "output_contract": "b010f9e78628265254ae5abef346f1c28d7851835aa1464e681b11dbe98dcf45",
    "claim_boundary": "f5aeb1f90bf03af6b4458160b79703f543ed28cbed56f8194006603fc32b8d5c",
}


class FinalTwellCompilerError(RuntimeError):
    """Raised when the final TWELL packet fails a frozen invariant."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def ordered_concept_ids() -> list[str]:
    return interim.ordered_concept_ids()


def receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_content_sha256", None)
    return content_sha256(payload)


def validate_config(config: Mapping[str, Any]) -> None:
    expected_sections = set(EXPECTED_SECTION_SEALS)
    if set(config) != expected_sections | {"section_seals"}:
        raise FinalTwellCompilerError("config top-level sections changed")
    seals = config["section_seals"]
    if set(seals) != expected_sections | {"unsealed_root_sha256"}:
        raise FinalTwellCompilerError("config seal inventory changed")
    for section, expected in EXPECTED_SECTION_SEALS.items():
        if seals[section] != expected or content_sha256(config[section]) != expected:
            raise FinalTwellCompilerError(f"sealed section changed: {section}")
    unsealed = {key: value for key, value in config.items() if key != "section_seals"}
    if (
        seals["unsealed_root_sha256"] != EXPECTED_UNSEALED_ROOT_SHA256
        or content_sha256(unsealed) != EXPECTED_UNSEALED_ROOT_SHA256
    ):
        raise FinalTwellCompilerError("unsealed config root changed")
    if content_sha256(config) != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise FinalTwellCompilerError("canonical config hash changed")
    identity = config["identity"]
    if (
        identity.get("packet_id") != PACKET_ID
        or identity.get("semantic_version") != "2.0.0"
        or identity.get("card_semantic_version") != "2.2.0"
        or identity.get("append_only") is not True
        or identity.get("frozen_before_response_access") is not True
    ):
        raise FinalTwellCompilerError("packet identity changed")
    bindings = list(config["hard_bindings"])
    binding_ids = [row["binding_id"] for row in bindings]
    if len(bindings) != 19 or len(binding_ids) != len(set(binding_ids)):
        raise FinalTwellCompilerError("hard-binding inventory changed")
    deferred = list(config["deferred_bindings"])
    if {row["binding_id"] for row in deferred} != {
        "SOURCE-AVAILABILITY-FINAL",
        "STATIC-RADIAL-ADAPTER-FINAL",
    }:
        raise FinalTwellCompilerError("deferred source inventory changed")
    if any(
        row.get("read_or_hashed") is not False
        or row.get("may_authorize_domain_readiness") is not False
        or "sha256" in row
        for row in deferred
    ):
        raise FinalTwellCompilerError("deferred source binding was hardened or authorized")
    compiler = config["compiler_contract"]
    if (
        compiler.get("total_count") != 400
        or compiler.get("expected_parameter_cell_count") != 1184
        or compiler.get("ordered_concept_ids_sha256") != content_sha256(ordered_concept_ids())
        or compiler.get("campaign_manifest_frozen") is not False
        or compiler.get("campaign_execution_authority") is not False
    ):
        raise FinalTwellCompilerError("compiler count or authority contract changed")
    lane_counts = config["lane_contract"]["expected_lane_counts"]
    if set(lane_counts) != set(registry.LANES) or sum(lane_counts.values()) != 400:
        raise FinalTwellCompilerError("five-lane contract changed")
    if any(config["access_contract"]["zero_access"].values()):
        raise FinalTwellCompilerError("zero-access contract changed")
    claim = config["claim_boundary"]
    if (
        claim.get("campaign_execution_authority") is not False
        or claim.get("campaign_manifest_frozen") is not False
        or claim.get("response_scoring_authorized") is not False
    ):
        raise FinalTwellCompilerError("final theory packet overclaims campaign authority")


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    try:
        config = json.loads((repo / CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalTwellCompilerError("could not load final TWELL config") from error
    if not isinstance(config, dict):
        raise FinalTwellCompilerError("final TWELL config must be an object")
    validate_config(config)
    return config


@dataclass
class MetadataLedger:
    repo: Path
    allowed: dict[Path, str]
    opened: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.allowed = {path.resolve(): kind for path, kind in self.allowed.items()}

    def read_bytes(self, path: Path) -> bytes:
        resolved = path.resolve()
        if resolved not in self.allowed:
            raise FinalTwellCompilerError(f"non-allowlisted metadata read refused: {resolved}")
        try:
            payload = resolved.read_bytes()
        except OSError as error:
            raise FinalTwellCompilerError(f"could not read metadata: {resolved}") from error
        try:
            display = resolved.relative_to(self.repo).as_posix()
        except ValueError:
            display = resolved.as_posix()
        self.opened[display] = self.allowed[resolved]
        return payload

    def rows(self) -> list[dict[str, str]]:
        return [{"path": path, "artifact_kind": self.opened[path]} for path in sorted(self.opened)]


def _path(repo: Path, text: str) -> Path:
    candidate = Path(text)
    return candidate if candidate.is_absolute() else repo / candidate


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalTwellCompilerError(f"invalid JSON: {label}") from error
    if not isinstance(value, dict):
        raise FinalTwellCompilerError(f"JSON object required: {label}")
    return value


def _jsonl_rows(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in payload.splitlines():
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("line is not an object")
                rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FinalTwellCompilerError(f"invalid canonical JSONL: {label}") from error
    return rows


def _binding_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["binding_id"]: row for row in config["hard_bindings"]}


def _bound_inputs(
    repo: Path, config: Mapping[str, Any]
) -> tuple[MetadataLedger, dict[str, bytes], bytes, bytes, bytes]:
    allowed = {
        repo / CONFIG_PATH: "final_twell_config",
        repo / MODULE_PATH: "final_twell_module",
        repo / TEST_PATH: "final_twell_tests",
    }
    for row in config["hard_bindings"]:
        allowed[_path(repo, row["path"])] = row["kind"]
    ledger = MetadataLedger(repo, allowed)
    raw_config = ledger.read_bytes(repo / CONFIG_PATH)
    module_bytes = ledger.read_bytes(repo / MODULE_PATH)
    test_bytes = ledger.read_bytes(repo / TEST_PATH)
    payloads: dict[str, bytes] = {}
    for row in config["hard_bindings"]:
        payload = ledger.read_bytes(_path(repo, row["path"]))
        if _sha256_bytes(payload) != row["sha256"]:
            raise FinalTwellCompilerError(f"hard binding changed: {row['binding_id']}")
        payloads[row["binding_id"]] = payload
    return ledger, payloads, raw_config, module_bytes, test_bytes


def _validate_bound_foundations(
    config: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    registry_receipt = _json_object(payloads["REGISTRY-RECEIPT-FINAL"], "registry receipt")
    mechanism_schema = _json_object(payloads["MECHANISM-CARD-SCHEMA-FINAL"], "mechanism schema")
    campaign_schema = _json_object(payloads["CAMPAIGN-MANIFEST-SCHEMA-FINAL"], "campaign schema")
    prior_receipt = _json_object(payloads["PRIOR-ART-v2-RECEIPT-FINAL"], "prior-art receipt")
    prior_payload = _json_object(payloads["PRIOR-ART-PAYLOAD-FINAL"], "prior-art payload")
    gp01_receipt = _json_object(payloads["GP01-RECEIPT-FINAL"], "GP01 receipt")
    preflight = _json_object(
        payloads["GP01-SOURCE-PREFLIGHT-RECEIPT-FINAL"], "GP01 source preflight"
    )
    interim_config = _json_object(payloads["INTERIM-TWELL-CONFIG"], "interim config")
    interim.validate_config(interim_config)
    registry_body = dict(registry_receipt)
    registry_claimed_hash = registry_body.pop("content_sha256", None)
    if registry_claimed_hash != registry.content_sha256(registry_body):
        raise FinalTwellCompilerError("registry receipt self-hash failed")
    if (
        registry_receipt["bindings"]["schemas"]["mechanism_card"]["file_sha256"]
        != (_binding_map(config)["MECHANISM-CARD-SCHEMA-FINAL"]["sha256"])
    ):
        raise FinalTwellCompilerError("registry/mechanism-schema binding mismatch")
    if registry.schema_errors({}, mechanism_schema) == []:
        raise FinalTwellCompilerError("mechanism schema unexpectedly accepts an empty card")
    if campaign_schema.get("$id") != "urn:invariant:open-gravity:campaign-manifest:1.0":
        raise FinalTwellCompilerError("campaign schema identity changed")
    if prior_receipt.get("decision") != (
        "PASS_FINAL_REGISTRY_REBOUND_PRIOR_ART_METADATA_ONLY_NO_CAMPAIGN_AUTHORITY"
    ):
        raise FinalTwellCompilerError("final prior-art receipt decision changed")
    prior_ids = {row["source_id"] for row in prior_payload["primary_sources"]}
    for architecture in interim_config["architecture_catalog"]:
        if not set(architecture["prior_art_source_ids"]) <= prior_ids:
            raise FinalTwellCompilerError(f"unknown prior-art source: {architecture['id']}")
    if gp01_receipt.get("decision") != (
        "GP01_FOUNDATION_PASS_SYNTHETIC_ONLY_ACTION_AND_CAUSAL_COMPLETION_QUARANTINED"
    ):
        raise FinalTwellCompilerError("GP01 receipt decision changed")
    if preflight.get("decision") != (
        "SOURCE_ONLY_PREFLIGHT_LOCAL_AND_ELLIPTIC_READY_T1_T2_BLOCKED_NO_Y100_ANCHOR"
    ):
        raise FinalTwellCompilerError("GP01 source-preflight decision changed")
    if preflight["adjudication"] != {
        "action_quarantined": 8,
        "aqual_spherical_equivalence_links": 8,
        "clusters_total": 8,
        "elliptic_source_ready_pending_solver": 8,
        "local_source_ready": 8,
        "missing_anchor_interpretation": "SOURCE_BLOCKED_NOT_EMPIRICAL_FAILURE",
        "telegraph_source_blocked": 8,
        "transport_source_blocked": 8,
        "transport_source_ready": 0,
    }:
        raise FinalTwellCompilerError("GP01 source-preflight adjudication changed")
    if any(
        row["nodes_at_or_above_y100"] != 0
        or row["transport_status"] != "SOURCE_BLOCKED_NO_UNIQUE_Y100_ANCHOR"
        for row in preflight["clusters"]
    ):
        raise FinalTwellCompilerError("GP01 no-anchor source fact changed")
    interim_rows = _jsonl_rows(payloads["INTERIM-TWELL-CARDS"], "interim cards")
    if [row.get("concept_id") for row in interim_rows] != ordered_concept_ids():
        raise FinalTwellCompilerError("interim concept order changed")
    for row in interim_rows:
        if row.get("card_sha256") != content_sha256(row["card"]):
            raise FinalTwellCompilerError(f"interim card self-hash failed: {row['concept_id']}")
        if row.get("probe_status") != "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES":
            raise FinalTwellCompilerError(f"interim exact probe failed: {row['concept_id']}")
    return interim_config, mechanism_schema, prior_payload, interim_rows


def _lane(concept_id: str, architecture_id: str) -> str:
    if concept_id.startswith("X"):
        return "ADJACENT" if int(concept_id[1:]) <= 10 else "WILDCARD"
    architecture_number = int(architecture_id[1:3])
    if architecture_number <= 7:
        return "CORE"
    if architecture_number <= 11:
        return "RIVALS_CONTROLS"
    return "ORTHOGONAL"


def _unit_string(unit_mapping: Mapping[str, Any]) -> str:
    return ";".join(f"{name}:{unit_mapping[name]}" for name in sorted(unit_mapping))


def _entries(
    interim_config: Mapping[str, Any],
) -> list[tuple[str, list[str], str, Mapping[str, Any] | None]]:
    drivers = {row["id"]: row for row in interim_config["driver_catalog"]}
    architectures = {row["id"]: row for row in interim_config["architecture_catalog"]}
    entries: list[tuple[str, list[str], str, Mapping[str, Any] | None]] = []
    for architecture_id in architectures:
        for driver_id in drivers:
            entries.append(
                (
                    f"TW2-{architecture_id.split('_')[0]}-{driver_id.split('_')[0]}",
                    [driver_id],
                    architecture_id,
                    None,
                )
            )
    for compound in interim_config["compound_catalog"]:
        entries.append(
            (compound["id"], list(compound["drivers"]), compound["architecture"], compound)
        )
    if [row[0] for row in entries] != ordered_concept_ids():
        raise FinalTwellCompilerError("final concept ordering changed")
    return entries


def compile_rows(
    config: Mapping[str, Any],
    interim_config: Mapping[str, Any],
    mechanism_schema: Mapping[str, Any],
    interim_rows: Sequence[Mapping[str, Any]],
    artifact_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    drivers = {row["id"]: row for row in interim_config["driver_catalog"]}
    architectures = {row["id"]: row for row in interim_config["architecture_catalog"]}
    driver_indexes = {driver_id: index for index, driver_id in enumerate(drivers, start=1)}
    predecessors = {row["concept_id"]: row for row in interim_rows}
    source_contract_sha256 = content_sha256(config["source_gate_contract"])
    rows: list[dict[str, Any]] = []
    for order_index, (concept_id, driver_ids, architecture_id, compound) in enumerate(
        _entries(interim_config)
    ):
        driver_rows = [drivers[driver_id] for driver_id in driver_ids]
        architecture = architectures[architecture_id]
        probe = interim._probe_entry(
            concept_id,
            driver_rows,
            architecture,
            compound,
            driver_indexes,
            interim_config,
        )
        if probe["status"] != "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES":
            raise FinalTwellCompilerError(f"exact-cell probe failed: {concept_id}")
        predecessor = predecessors[concept_id]
        card = interim._card(
            concept_id,
            driver_rows,
            architecture,
            compound,
            probe,
            artifact_hashes,
            predecessor["card_sha256"],
        )
        for cell in card["parameter_cells"]:
            cell["unit"] = _unit_string(cell["unit"])
        card["card_id"] = f"{concept_id}@2.2.0"
        card["semantic_version"] = "2.2.0"
        card["parents"] = [
            {
                "card_id": predecessor["card"]["card_id"],
                "card_sha256": predecessor["card_sha256"],
                "relation": "REPAIRS",
            }
        ]
        card["author_agent"] = "target-blind-twell-400-v2-final-hard-bound-compiler"
        card["provenance"]["origin_artifacts"] = [
            PACKET_ID,
            "TWELL-400-v2",
            "GRAVITY-LIGHT-GRAMMAR-v1",
            "OPEN-GRAVITY-PRIMARY-SOURCE-COMPARATOR-CONTRACT-v2",
            "GAIN-PERSISTENCE-01",
        ]
        card["version_change"] = {
            "kind": "MINOR",
            "previous_card_id": predecessor["card"]["card_id"],
            "previous_card_sha256": predecessor["card_sha256"],
            "changed_facets": [
                "configuration_sha256",
                "mechanism_schema_binding",
                "parameter_cell_unit_encoding",
                "registry_binding",
                "source_gate_disposition",
            ],
            "prior_result_retained": True,
            "replay_all_affected": True,
        }
        card["hashes"]["code_sha256"] = artifact_hashes["code_sha256"]
        card["hashes"]["data_sha256"] = artifact_hashes["data_sha256"]
        card["hashes"]["environment_sha256"] = artifact_hashes["environment_sha256"]
        card["hashes"]["configuration_sha256"] = artifact_hashes["configuration_sha256"]
        card["hashes"]["formula_sha256"] = registry.mechanism_formula_sha256(card)
        admission = registry.mechanism_card_admission(card, mechanism_schema)
        if admission != {"eligible": True, "status": "READY_FOR_THEORY_GATES", "errors": []}:
            raise FinalTwellCompilerError(
                f"final mechanism card is not registry-admissible: {concept_id}:{admission}"
            )
        lane = _lane(concept_id, architecture_id)
        equivalence_family_id = f"EQ-{card['hashes']['formula_sha256'][:24]}"
        domain_execution = {
            domain: {
                "eligible": False,
                "execution_disposition": "THEORY_ONLY",
                "scored": False,
                "source_contract_sha256": source_contract_sha256,
            }
            for domain in registry.DOMAINS
        }
        manifest_input = {
            "candidate_id": concept_id,
            "card_id": card["card_id"],
            "semantic_version": card["semantic_version"],
            "anonymous_formula_id": f"F{order_index + 1:04d}",
            "lane": lane,
            "candidate_status": "REGISTERED_THEORY_ONLY",
            "scientific_status": card["scientific_status"],
            "identity_class": card["identity_class"],
            "mechanism_kind": card["action_or_equations"]["kind"],
            "mechanism_executable": card["action_or_equations"]["executable"],
            "card_sha256": content_sha256(card),
            "formula_sha256": card["hashes"]["formula_sha256"],
            "configuration_sha256": card["hashes"]["configuration_sha256"],
            "equivalence_family_id": equivalence_family_id,
            "equivalence_fingerprint_sha256": registry.equivalence_fingerprint_sha256(card),
            "domain_execution": domain_execution,
        }
        rows.append(
            {
                "concept_id": concept_id,
                "order_index": order_index,
                "entry_kind": "ATOMIC" if compound is None else "COMPOUND",
                "driver_ids": driver_ids,
                "architecture_id": architecture_id,
                "lane": lane,
                "compiler_status": "REGISTERED_THEORY_ONLY_PENDING_FINAL_SOURCE_MATRIX",
                "schema_admission": admission,
                "probe_status": probe["status"],
                "execution_class": probe["execution_class"],
                "parameter_cell_count": probe["parameter_cell_count"],
                "passed_parameter_cell_count": probe["passed_parameter_cell_count"],
                "failed_parameter_cell_count": probe["failed_parameter_cell_count"],
                "maximum_computed_operator_residual": probe["maximum_computed_operator_residual"],
                "maximum_computed_boundary_or_initial_residual": probe[
                    "maximum_computed_boundary_or_initial_residual"
                ],
                "maximum_computed_analytic_lambda_zero_residual": probe[
                    "maximum_computed_analytic_lambda_zero_residual"
                ],
                "cell_results": probe["cell_results"],
                "probe_digest_sha256": probe["fixture_digest_sha256"],
                "equivalence_family_id": equivalence_family_id,
                "card_sha256": content_sha256(card),
                "card": card,
                "manifest_input": manifest_input,
            }
        )
    return rows


def cards_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def stream_root(rows: Sequence[Mapping[str, Any]]) -> str:
    return content_sha256([_sha256_bytes(_canonical(row)) for row in rows])


def build_packet(root: Path | None = None) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    repo = _repo_root() if root is None else root.resolve()
    config = load_config(repo)
    ledger, payloads, raw_config, module_bytes, test_bytes = _bound_inputs(repo, config)
    interim_config, mechanism_schema, prior_payload, interim_rows = _validate_bound_foundations(
        config, payloads
    )
    binding_hashes = {row["binding_id"]: row["sha256"] for row in config["hard_bindings"]}
    artifact_hashes = {
        "code_sha256": _sha256_bytes(module_bytes),
        "data_sha256": content_sha256(
            {
                "prior_art": binding_hashes["PRIOR-ART-v2-RECEIPT-FINAL"],
                "gp01": binding_hashes["GP01-RECEIPT-FINAL"],
                "source_preflight": binding_hashes["GP01-SOURCE-PREFLIGHT-RECEIPT-FINAL"],
                "source_matrix": "DEFERRED_NOT_READ_NOT_HASH_BOUND",
            }
        ),
        "environment_sha256": content_sha256(
            "CPython>=3.11;stdlib-only;IEEE754-binary64;canonical-json-sort-keys-ascii"
        ),
        "configuration_sha256": _sha256_bytes(raw_config),
    }
    rows = compile_rows(config, interim_config, mechanism_schema, interim_rows, artifact_hashes)
    payload = cards_bytes(rows)
    cards = [row["card"] for row in rows]
    candidates = [row["manifest_input"] for row in rows]
    cell_count = sum(row["parameter_cell_count"] for row in rows)
    if cell_count != config["compiler_contract"]["expected_parameter_cell_count"]:
        raise FinalTwellCompilerError("final parameter-cell count changed")
    lane_counts = dict(sorted(Counter(row["lane"] for row in rows).items()))
    if lane_counts != config["lane_contract"]["expected_lane_counts"]:
        raise FinalTwellCompilerError("final lane partition changed")
    formula_families: defaultdict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        formula_families[candidate["formula_sha256"]].add(candidate["equivalence_family_id"])
    if any(len(families) != 1 for families in formula_families.values()):
        raise FinalTwellCompilerError("identical formulas split across equivalence families")
    receipt: dict[str, Any] = {
        "schema_version": "invariant-twell-400-v2-typed-compiler-final-receipt-2.0",
        "packet_id": PACKET_ID,
        "semantic_version": "2.0.0",
        "decision": DECISION,
        "status": "MUTATION_FROZEN_FINAL_THEORY_PACKET_PENDING_CLEAN_AUDIT",
        "enumeration": {
            "atomic_count": 380,
            "compound_count": 20,
            "total_count": len(rows),
            "parameter_cell_count": cell_count,
            "cartesian_cell_count": sum(
                cell["cell_kind"] == "CARTESIAN" for row in rows for cell in row["cell_results"]
            ),
            "compound_override_evidence_count": sum(
                cell["cell_kind"] == "COMPOUND_OVERRIDE_EVIDENCE"
                for row in rows
                for cell in row["cell_results"]
            ),
            "ordered_concept_ids_sha256": content_sha256([row["concept_id"] for row in rows]),
        },
        "exact_cell_evidence": {
            "passed_card_count": sum(
                row["probe_status"] == "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES" for row in rows
            ),
            "passed_cell_count": sum(row["passed_parameter_cell_count"] for row in rows),
            "failed_cell_count": sum(row["failed_parameter_cell_count"] for row in rows),
            "maximum_computed_operator_residual": max(
                row["maximum_computed_operator_residual"] for row in rows
            ),
            "maximum_computed_boundary_or_initial_residual": max(
                row["maximum_computed_boundary_or_initial_residual"] for row in rows
            ),
            "maximum_computed_analytic_lambda_zero_residual": max(
                row["maximum_computed_analytic_lambda_zero_residual"] for row in rows
            ),
        },
        "registry_manifest_input": {
            "complete_live_card_count": len(cards),
            "schema_valid_card_count": sum(
                not registry.schema_errors(card, mechanism_schema) for card in cards
            ),
            "mechanism_card_set_sha256": registry.mechanism_card_set_sha256(cards),
            "candidate_version_count": len(candidates),
            "candidate_versions_sha256": content_sha256(candidates),
            "equivalence_ledger_sha256": registry.campaign_equivalence_ledger_sha256(candidates),
            "lane_counts": lane_counts,
            "all_five_lanes_present": set(lane_counts) == set(registry.LANES),
            "candidate_status_counts": dict(
                sorted(Counter(row["candidate_status"] for row in candidates).items())
            ),
            "domain_disposition_counts": {
                domain: dict(
                    sorted(
                        Counter(
                            row["domain_execution"][domain]["execution_disposition"]
                            for row in candidates
                        ).items()
                    )
                )
                for domain in registry.DOMAINS
            },
            "campaign_manifest_frozen": False,
            "manifest_execution_authority": False,
        },
        "theory_only_catalogs": {
            "qg_ontology_rows": len(prior_payload["ontology_prior_art"]),
            "light_gravity_analogy_rows": len(prior_payload["light_gravity_analogies"]),
            "rival_comparator_rows": len(prior_payload["dynamical_comparators"]),
            "catalog_rows_are_not_duplicated_or_scored_cards": True,
        },
        "source_gate": {
            **dict(config["source_gate_contract"]),
            "source_contract_sha256": content_sha256(config["source_gate_contract"]),
            "deferred_bindings": list(config["deferred_bindings"]),
        },
        "stream": {
            "path": CARDS_PATH.as_posix(),
            "file_sha256": _sha256_bytes(payload),
            "ordered_line_root_sha256": stream_root(rows),
            "line_count": len(rows),
            "format": "canonical-jsonl",
        },
        "predecessors": {
            "blocked_original_cards_sha256": binding_hashes["BLOCKED-ORIGINAL-TWELL-CARDS"],
            "blocked_original_receipt_sha256": binding_hashes["BLOCKED-ORIGINAL-TWELL-RECEIPT"],
            "interim_cards_sha256": binding_hashes["INTERIM-TWELL-CARDS"],
            "interim_receipt_sha256": binding_hashes["INTERIM-TWELL-RECEIPT"],
            "both_predecessors_retained": True,
        },
        "semantic_seals": dict(config["section_seals"]),
        "artifact_hashes": {
            "config_file_sha256": _sha256_bytes(raw_config),
            "config_canonical_sha256": content_sha256(config),
            "module_sha256": _sha256_bytes(module_bytes),
            "test_sha256": _sha256_bytes(test_bytes),
            "hard_bindings": binding_hashes,
        },
        "access_audit": {
            **dict(config["access_contract"]["zero_access"]),
            "deferred_source_or_adapter_files_opened": 0,
            "astronomy_response_payloads_opened": 0,
            "allowlisted_metadata_files_opened": ledger.rows(),
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    receipt["receipt_content_sha256"] = receipt_content_sha256(receipt)
    return rows, payload, receipt


def _temp_payload(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _rollback_owned_hardlink(temporary: Path, destination: Path) -> None:
    try:
        owned = os.path.samestat(temporary.stat(), destination.stat())
    except FileNotFoundError:
        return
    except OSError as error:
        raise FinalTwellCompilerError("could not prove partial-publication ownership") from error
    if not owned:
        raise FinalTwellCompilerError("refusing to remove a non-owned card stream")
    destination.unlink(missing_ok=True)


def _atomic_packet_no_clobber(
    cards_path: Path, cards_payload: bytes, receipt_path: Path, receipt_payload: bytes
) -> None:
    if cards_path.exists() or receipt_path.exists():
        raise FinalTwellCompilerError("refusing to overwrite final TWELL packet")
    cards_temp = _temp_payload(cards_path, cards_payload)
    receipt_temp = _temp_payload(receipt_path, receipt_payload)
    cards_linked = False
    try:
        os.link(cards_temp, cards_path)
        cards_linked = True
        os.link(receipt_temp, receipt_path)
    except FileExistsError as error:
        if cards_linked:
            _rollback_owned_hardlink(cards_temp, cards_path)
        raise FinalTwellCompilerError("refusing to overwrite final TWELL packet") from error
    except OSError as error:
        if cards_linked:
            _rollback_owned_hardlink(cards_temp, cards_path)
        raise FinalTwellCompilerError("atomic final TWELL publication failed") from error
    finally:
        cards_temp.unlink(missing_ok=True)
        receipt_temp.unlink(missing_ok=True)


def write_packet(
    cards_payload: bytes, receipt: Mapping[str, Any], cards_path: Path, receipt_path: Path
) -> None:
    receipt_payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    _atomic_packet_no_clobber(cards_path, cards_payload, receipt_path, receipt_payload)


def check_packet(
    root: Path | None = None,
    cards_path: Path | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    cards_target = repo / CARDS_PATH if cards_path is None else cards_path.resolve()
    receipt_target = repo / RECEIPT_PATH if receipt_path is None else receipt_path.resolve()
    _rows, expected_cards, expected_receipt = build_packet(repo)
    try:
        actual_cards = cards_target.read_bytes()
        actual_receipt = json.loads(receipt_target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalTwellCompilerError("could not read stored final TWELL packet") from error
    if actual_cards != expected_cards or actual_receipt != expected_receipt:
        raise FinalTwellCompilerError("stored final TWELL packet differs from rebuild")
    if actual_receipt["receipt_content_sha256"] != receipt_content_sha256(actual_receipt):
        raise FinalTwellCompilerError("stored final TWELL receipt self-hash failed")
    return actual_receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--cards", type=Path, default=CARDS_PATH)
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    args = parser.parse_args(argv)
    repo = _repo_root()
    try:
        if args.command == "build":
            _rows, payload, receipt = build_packet(repo)
            cards_path = args.cards if args.cards.is_absolute() else repo / args.cards
            receipt_path = args.receipt if args.receipt.is_absolute() else repo / args.receipt
            write_packet(payload, receipt, cards_path, receipt_path)
        else:
            cards_path = args.cards if args.cards.is_absolute() else repo / args.cards
            receipt_path = args.receipt if args.receipt.is_absolute() else repo / args.receipt
            receipt = check_packet(repo, cards_path, receipt_path)
    except (FinalTwellCompilerError, registry.OpenGravityRegistryError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "total_count": receipt["enumeration"]["total_count"],
                "parameter_cell_count": receipt["enumeration"]["parameter_cell_count"],
                "receipt_content_sha256": receipt["receipt_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
