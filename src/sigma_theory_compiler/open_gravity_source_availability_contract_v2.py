"""Append-only source-availability successor for the open-gravity registry.

The module reads only its metadata contracts.  It streams the exact 65,100
mechanism/object/observable dispositions, plus manifest-oriented concept and
parameter-cell domain projections, without opening an astronomy response or
an upstream receipt payload.  It grants neither ``DATA_ELIGIBLE`` nor campaign
execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_source_availability_contract_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_source_availability_contract_v2.py")
TEST_PATH = Path("tests/test_open_gravity_source_availability_contract_v2.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-source-availability-contract-v2.json")

_CANONICAL_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_source_availability_contract_v2.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_source_availability_contract_v2.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_source_availability_contract_v2.py")
_CANONICAL_OUTPUT_PATH = Path("runs/gravity/open-gravity-source-availability-contract-v2.json")

SCHEMA = "invariant-open-gravity-source-availability-contract-2.0"
RECEIPT_SCHEMA = "invariant-open-gravity-source-availability-receipt-2.0"
SOURCE_CONTRACT_SCHEMA = "invariant-open-gravity-domain-source-contract-1.0"
EXPECTED_TWELL_IDS_SHA256 = "7388f8982c5014ef6c365d00aa780ba2ecb8b8b3f6786658fb3db36b64c29c5f"
EXPECTED_CONFIG_FILE_SHA256 = "5192fe16d77760907c0be24358f95b8ac1be3fb2f37c7520bcfadb76a91f4543"
EXPECTED_CONFIG_CONTENT_SHA256 = "527a03b903be304f1aa6d923cb33be525d92072738112f523e0043eb0f211c66"
EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256 = "594666bc1008fef4207ff54a1a2a9c2cfa570573d7f7b963da7d086bcd66829d"  # fmt: skip
EXPECTED_TEST_FILE_SHA256 = "902d9ddf737f1f0831c85bb0e8fbf3fc257066f93520340263414bc75490aca5"
EXPECTED_TEST_SEMANTIC_SHA256 = "902d9ddf737f1f0831c85bb0e8fbf3fc257066f93520340263414bc75490aca5"
FINAL_GATE_STATUS = "BOUND_FINAL_AGENT_REPORTED_HASHES_NO_RECEIPT_PAYLOAD_OPENED"
PENDING_GATE_STATUS = "PENDING_FINAL_AGENT_REPORTED_HASHES"
BLOCKED_GATE_STATUS = "BLOCKED_AUDIT_FAILED_REPLACEMENT_REQUIRED"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

STRICT_SECTIONS = (
    "authority_boundary",
    "predecessor_counterevidence",
    "inherited_sections",
    "incident_enforcement",
    "committed_bindings",
    "final_bind_gates",
    "disposition_contract",
    "gp01_dispositions",
    "ontology_disposition",
    "comparator_policy",
    "discriminator_source_block",
    "manifest_projection_contract",
    "matrix_contract",
    "claim_boundary",
)
REGISTRY_DOMAINS = ("GALAXIES", "GROUPS", "CLUSTERS", "LENSING")
_IMPLEMENTATION_SEMANTIC_PIN_NAME = "EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256"
_IMPLEMENTATION_PIN_RE = re.compile(
    rf'^{_IMPLEMENTATION_SEMANTIC_PIN_NAME} = "[^"]+"(?:  # fmt: skip)?$',
    re.MULTILINE,
)


class SourceAvailabilityV2Error(RuntimeError):
    """Raised when the v2 metadata-only source contract fails closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def text_semantic_sha256(payload: bytes) -> str:
    try:
        normalized = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise SourceAvailabilityV2Error("artifact is not UTF-8 text") from exc
    return bytes_sha256(normalized.encode("utf-8"))


def implementation_semantic_sha256(payload: bytes) -> str:
    try:
        normalized = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise SourceAvailabilityV2Error("implementation is not UTF-8 text") from exc
    matches = list(_IMPLEMENTATION_PIN_RE.finditer(normalized))
    _require(len(matches) == 1, "implementation semantic pin assignment changed")
    normalized = _IMPLEMENTATION_PIN_RE.sub(
        f'{_IMPLEMENTATION_SEMANTIC_PIN_NAME} = "<SELF_PIN>"  # fmt: skip',
        normalized,
    )
    return bytes_sha256(normalized.encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAvailabilityV2Error(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAvailabilityV2Error(f"cannot read {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _canonical_repo_path(current: Path, expected: Path, label: str) -> Path:
    _require(current == expected, f"canonical {label} path changed")
    path = (_CANONICAL_REPO_ROOT / expected).resolve()
    _require(
        path.is_relative_to(_CANONICAL_REPO_ROOT),
        f"canonical {label} path escaped repository",
    )
    return path


def _canonical_artifact_paths() -> tuple[Path, Path, Path]:
    return (
        _canonical_repo_path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config"),
        _canonical_repo_path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module"),
        _canonical_repo_path(TEST_PATH, _CANONICAL_TEST_PATH, "test"),
    )


def _canonical_artifact_bytes() -> tuple[bytes, bytes, bytes]:
    config_path, module_path, test_path = _canonical_artifact_paths()
    return config_path.read_bytes(), module_path.read_bytes(), test_path.read_bytes()


def _canonical_output_path() -> Path:
    return _canonical_repo_path(OUTPUT_PATH, _CANONICAL_OUTPUT_PATH, "output")


def _decode_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAvailabilityV2Error(f"cannot decode {label}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def validate_artifact_integrity(
    config_bytes: bytes, module_bytes: bytes, test_bytes: bytes
) -> dict[str, Any]:
    _require(
        bytes_sha256(config_bytes) == EXPECTED_CONFIG_FILE_SHA256,
        "v2 config raw file hash changed",
    )
    config = _decode_json_bytes(config_bytes, "v2 source contract")
    _require(
        content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256,
        "v2 config semantic hash changed",
    )
    _require(
        implementation_semantic_sha256(module_bytes) == EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256,
        "v2 implementation semantic hash changed",
    )
    _require(
        bytes_sha256(test_bytes) == EXPECTED_TEST_FILE_SHA256,
        "v2 test raw file hash changed",
    )
    _require(
        text_semantic_sha256(test_bytes) == EXPECTED_TEST_SEMANTIC_SHA256,
        "v2 test semantic hash changed",
    )
    validate_config(config)
    return config


def twell_concept_ids() -> list[str]:
    return [
        f"TW2-A{architecture:02d}-D{driver:02d}"
        for architecture in range(1, 20)
        for driver in range(1, 21)
    ] + [f"X{compound:02d}" for compound in range(1, 21)]


def mechanism_catalog(predecessor: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = predecessor["mechanism_registry"]
    twell = registry["twell"]
    catalog: list[dict[str, Any]] = []
    for architecture_index, architecture in enumerate(twell["architectures"], start=1):
        for driver_index, driver in enumerate(twell["drivers"], start=1):
            catalog.append(
                {
                    "mechanism_id": f"TW2-A{architecture_index:02d}-D{driver_index:02d}",
                    "mechanism_family": "TWELL_ATOMIC",
                    "discovery_lane": "CORE",
                    "drivers": [driver],
                    "architecture": architecture,
                }
            )
    for compound in twell["compounds"]:
        catalog.append(
            {
                "mechanism_id": compound["id"],
                "mechanism_family": "TWELL_COMPOUND",
                "discovery_lane": "CORE",
                "drivers": list(compound["drivers"]),
                "architecture": compound["architecture"],
            }
        )
    for variant in registry["GP01_variants"]:
        catalog.append(
            {
                "mechanism_id": variant,
                "mechanism_family": "GP01",
                "discovery_lane": "CORE",
                "drivers": [],
                "architecture": None,
            }
        )
    for ontology in registry["ontology_nodes"]:
        lane = "RIVALS_CONTROLS" if ontology == "QG01" else "ORTHOGONAL"
        if ontology == "QG13":
            lane = "WILDCARD"
        catalog.append(
            {
                "mechanism_id": ontology,
                "mechanism_family": "GRAVITY_LIGHT_ONTOLOGY",
                "discovery_lane": lane,
                "drivers": [],
                "architecture": None,
            }
        )
    return catalog


def _all_final_gates_bound(config: Mapping[str, Any]) -> bool:
    return all(
        gate.get("status") == FINAL_GATE_STATUS for gate in config["final_bind_gates"].values()
    )


def _gate_bound(config: Mapping[str, Any], gate_id: str) -> bool:
    return config["final_bind_gates"][gate_id]["status"] == FINAL_GATE_STATUS


def require_final_gates(config: Mapping[str, Any]) -> None:
    pending = [
        gate_id
        for gate_id, gate in config["final_bind_gates"].items()
        if gate["status"] != FINAL_GATE_STATUS
    ]
    _require(not pending, "final bind gates remain pending: " + ", ".join(pending))


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == SCHEMA, "v2 schema changed")
    _require(
        config.get("status")
        == "SEALED_FINAL_DEPENDENCIES_AUDITED_ZERO_RESPONSE_ACCESS_NO_CAMPAIGN_AUTHORITY",
        "v2 seal status changed",
    )
    _require(config.get("append_only") is True, "append-only policy changed")
    _require(
        config.get("output_path") == _CANONICAL_OUTPUT_PATH.as_posix(),
        "output path changed",
    )
    _require(
        set(config.get("section_sha256", {})) == set(STRICT_SECTIONS), "section seal set changed"
    )
    for section in STRICT_SECTIONS:
        observed = content_sha256(config[section])
        expected = config["section_sha256"][section]
        _require(observed == expected, f"strict section hash changed: {section}")
    if EXPECTED_CONFIG_CONTENT_SHA256 != "PENDING_FINAL_BINDINGS":
        _require(
            content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256,
            "v2 config semantics changed",
        )

    authority = config["authority_boundary"]
    for key in (
        "may_claim_DATA_ELIGIBLE",
        "may_freeze_campaign_manifest",
        "may_authorize_response_execution",
        "may_open_or_score_responses",
        "may_choose_repair_or_rank_candidates",
        "may_adjudicate_prior_art_or_novelty",
        "campaign_authority_granted",
    ):
        _require(authority[key] is False, f"forbidden authority enabled: {key}")
    predecessor = config["predecessor_counterevidence"]
    _require(predecessor["preserved_unchanged"] is True, "predecessor not preserved")
    _require(predecessor["receipt_payload_opened_by_v2_generator"] is False, "receipt opened")
    _require(config["incident_enforcement"]["retained_incident_count"] == 2, "incidents changed")

    committed = {row["binding_id"]: row for row in config["committed_bindings"]}
    _require(
        committed["OPEN_GRAVITY_REGISTRY_FOUNDATION"]["commit"]
        == "74cf64129787163cbead8dccb243fa4faf86fbe1",
        "registry commit changed",
    )
    _require(
        committed["GP01_FOUNDATION"]["commit"] == "35f70938f158c81971b2e1b838371b09d9fcee2c",
        "GP01 commit changed",
    )
    preflight = committed["GP01_XCOP_SOURCE_PREFLIGHT"]
    _require(
        preflight["commit"] == "ed2988546fb1165d9efe5e62d52cddebc7b1a79d",
        "GP01 X-COP commit changed",
    )
    _require(
        preflight["receipt_decision"]
        == "SOURCE_ONLY_PREFLIGHT_LOCAL_AND_ELLIPTIC_READY_T1_T2_BLOCKED_NO_Y100_ANCHOR",
        "GP01 X-COP PASS decision changed",
    )
    for binding in config["committed_bindings"]:
        _require(SHA256_RE.fullmatch(binding["commit"]) is None, "commit mislabeled as sha256")
        for artifact in binding["artifacts"]:
            _require(SHA256_RE.fullmatch(artifact["sha256"]) is not None, "binding hash malformed")

    gates = config["final_bind_gates"]
    _require(
        set(gates) == {"TWELL_SUCCESSOR", "PRIMARY_SOURCE_PRIOR_ART", "STATIC_RADIAL_ADAPTER"},
        "final bind gate set changed",
    )
    for gate in gates.values():
        _require(
            gate["status"] in {PENDING_GATE_STATUS, BLOCKED_GATE_STATUS, FINAL_GATE_STATUS},
            "invalid gate status",
        )
        for ledger_key in ("failed_predecessor_artifacts", "artifacts"):
            artifacts = gate.get(ledger_key, [])
            _require(
                len({row["role"] for row in artifacts}) == len(artifacts),
                f"duplicate gate artifact role: {ledger_key}",
            )
            for artifact in artifacts:
                _require(
                    SHA256_RE.fullmatch(artifact["sha256"]) is not None,
                    f"gate hash malformed: {ledger_key}",
                )
                for hash_key in ("canonical_sha256", "semantic_sha256", "content_sha256"):
                    if hash_key in artifact:
                        _require(
                            SHA256_RE.fullmatch(artifact[hash_key]) is not None,
                            f"gate {hash_key} malformed: {ledger_key}",
                        )
        if gate["status"] == BLOCKED_GATE_STATUS:
            _require(gate.get("replacement_required") is True, "blocked gate needs replacement")
            _require(gate["artifacts"] == [], "blocked artifacts were left active")
            _require(
                {row["role"] for row in gate["failed_predecessor_artifacts"]}
                == set(gate["required_roles"]),
                "failed predecessor artifact roles changed",
            )
            _require(
                str(gate.get("independent_audit_status", "")).startswith("BLOCKED_"),
                "blocked audit status changed",
            )
            _require(gate.get("independent_audit_findings"), "blocked audit findings missing")
        if gate["status"] == FINAL_GATE_STATUS:
            _require(
                gate.get("replacement_required") is False, "final gate still needs replacement"
            )
            _require(
                gate.get("independent_audit_status")
                == gate.get("required_independent_audit_status"),
                "final gate lacks required independent audit PASS",
            )
            _require(
                COMMIT_RE.fullmatch(str(gate.get("commit", ""))) is not None,
                "final gate commit malformed",
            )
            roles = {row["role"] for row in gate["artifacts"]}
            _require(roles == set(gate["required_roles"]), "final gate artifact roles changed")
            for artifact in gate["artifacts"]:
                _require(SHA256_RE.fullmatch(artifact["sha256"]) is not None, "gate hash malformed")

    disposition = config["disposition_contract"]
    _require(
        len(disposition["static_ready_architectures"]) == 15, "static ready architectures changed"
    )
    _require(
        disposition["static_blocked_architectures"]
        == ["A15_RETARDED", "A16_MEMORY", "A17_RESONANCE", "A18_STOCHASTIC"],
        "static blocked architectures changed",
    )
    _require(
        disposition["domain_driver_allowlists"]["SPARC"]
        == ["D01_ACC", "D03_RAD", "D06_SLOPE", "D13_GASF"],
        "SPARC driver partition changed",
    )
    _require(len(disposition["domain_driver_allowlists"]["XCOP"]) == 8, "X-COP drivers changed")
    _require(
        disposition["domain_compound_allowlists"]["XCOP"]
        == ["X01", "X05", "X10", "X13", "X17", "X18"],
        "X-COP compounds changed",
    )
    _require(disposition["domain_compound_allowlists"]["SPARC"] == [], "SPARC compound leak")
    _require(
        disposition["source_ready_concept_counts"] == {"SPARC_TWELL": 60, "XCOP_TWELL": 126},
        "TWELL concept counts changed",
    )
    _require(
        disposition["source_ready_parameter_cell_counts"]
        == {"SPARC_TWELL": 176, "XCOP_TWELL": 370},
        "TWELL cell counts changed",
    )
    _require(
        set(config["gp01_dispositions"])
        == {
            "GP01-L",
            "GP01-AQUAL",
            "GP01-T1",
            "GP01-T2",
            "GP01-ELLIPTIC",
            "GP01-TELEGRAPH",
            "GP01-ACTION_PLACEHOLDER",
        },
        "GP01 variants changed",
    )
    _require(len(config["ontology_disposition"]["nodes"]) == 13, "ontology nodes changed")

    matrix = config["matrix_contract"]
    _require(matrix["mechanisms"] == 420, "mechanism count changed")
    _require(matrix["object_observable_slots"] == 155, "observable slot count changed")
    _require(matrix["expanded_tuple_count"] == 65100, "matrix count changed")
    _require(
        matrix["stream_only"] is True and matrix["rows_materialized_in_receipt"] == 0,
        "matrix materialized",
    )
    projection = config["manifest_projection_contract"]
    _require(projection["registry_domains"] == list(REGISTRY_DOMAINS), "registry domains changed")
    _require(projection["concept_domain_slot_count"] == 1680, "concept domain slots changed")
    _require(
        projection["exact_parameter_cell_sources"]["total"] == 2486, "cell source count changed"
    )
    _require(
        projection["exact_parameter_cell_sources"]["domain_slot_count"] == 9944,
        "cell domain slots changed",
    )
    _require(
        projection["source_ready_registry_projection"]["eligible"] is False, "eligibility claimed"
    )
    _require(config["claim_boundary"]["DATA_ELIGIBLE_claimed"] is False, "DATA_ELIGIBLE claimed")
    _require(
        config["claim_boundary"]["campaign_execution_authorized"] is False, "campaign authorized"
    )


def _bound_repo_path(relative: str, label: str) -> Path:
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"{label} path must be repository-relative")
    path = (_CANONICAL_REPO_ROOT / relative_path).resolve()
    _require(path.is_relative_to(_CANONICAL_REPO_ROOT), f"{label} path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return validate_artifact_integrity(*_canonical_artifact_bytes())


def load_predecessor(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["predecessor_counterevidence"]
    path = _bound_repo_path(binding["path"], "predecessor")
    _require(file_sha256(path) == binding["file_sha256"], "predecessor file changed")
    value = _read_json(path, "v1 predecessor metadata")
    _require(
        content_sha256(value) == binding["semantic_content_sha256"], "predecessor semantics changed"
    )
    for section, expected in config["inherited_sections"].items():
        _require(
            content_sha256(value[section]) == expected, f"inherited section changed: {section}"
        )
    return value


def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config()
    return config, load_predecessor(config)


_STATUS_PRECEDENCE = {
    "FORBIDDEN_RESPONSE_DERIVATION": 90,
    "UNKNOWN_SOURCE_BLOCKED": 80,
    "SOURCE_MISSING": 70,
    "SOURCE_INCOMPLETE": 60,
    "SOURCE_AVAILABLE_DERIVED_SPHERICAL_MODEL_ONLY": 50,
    "SOURCE_AVAILABLE_WITH_SHARED_GLOBAL_STELLAR_NUISANCE": 40,
    "SOURCE_AVAILABLE_SPHERICAL_RADIAL_ONLY": 30,
    "SOURCE_AVAILABLE_SHARED_MEASUREMENT_ANCESTRY": 20,
    "SOURCE_AVAILABLE": 10,
    "OUT_OF_SCOPE_NOT_OPENED": 0,
}


def _combined_driver_status(
    predecessor: Mapping[str, Any], domain: str, drivers: Sequence[str]
) -> str:
    if not drivers:
        return "OUT_OF_SCOPE_NOT_OPENED"
    statuses = [predecessor["driver_source_availability"][domain][driver] for driver in drivers]
    return max(statuses, key=_STATUS_PRECEDENCE.__getitem__)


def _twell_disposition(
    config: Mapping[str, Any], mechanism: Mapping[str, Any], domain: str
) -> tuple[str, str, int, int, str]:
    rules = config["disposition_contract"]
    mechanism_id = mechanism["mechanism_id"]
    architecture = mechanism["architecture"]
    declared = int(rules["architecture_parameter_cell_counts"][architecture])
    if mechanism_id in {"X19", "X20"}:
        declared += 1
    if architecture in rules["static_blocked_architectures"]:
        return (
            "SOURCE_BLOCKED",
            "SOURCE_BLOCKED_NO_PARAMETER_CELL_EXECUTION",
            declared,
            0,
            f"{architecture}_BLOCKED_ON_STATIC_PROFILES",
        )
    if mechanism["mechanism_family"] == "TWELL_COMPOUND":
        if domain == "SPARC" and mechanism_id == "X01":
            reason = rules["SPARC_X01_explicit_block"]
        else:
            reason = "COMPOUND_NOT_IN_EXACT_DOMAIN_ALLOWLIST"
        if mechanism_id not in rules["domain_compound_allowlists"][domain]:
            return (
                "SOURCE_BLOCKED",
                "SOURCE_BLOCKED_NO_PARAMETER_CELL_EXECUTION",
                declared,
                0,
                reason,
            )
    elif not set(mechanism["drivers"]) <= set(rules["domain_driver_allowlists"][domain]):
        return (
            "SOURCE_BLOCKED",
            "SOURCE_BLOCKED_NO_PARAMETER_CELL_EXECUTION",
            declared,
            0,
            "DRIVER_OUTSIDE_EXACT_DOMAIN_SOURCE_ALLOWLIST",
        )
    concept = (
        "SOURCE_READY_STATIC_RADIAL_CONCEPT"
        if domain == "SPARC"
        else "SOURCE_READY_STATIC_SPHERICAL_RADIAL_CONCEPT"
    )
    if _gate_bound(config, "TWELL_SUCCESSOR"):
        parameter = "SOURCE_READY_COMPILED_PARAMETER_CELLS_NO_DATA_ELIGIBILITY"
        sealed = declared
    else:
        parameter = "PENDING_FINAL_TWELL_SUCCESSOR_BIND"
        sealed = 0
    return concept, parameter, declared, sealed, "STATIC_ARCHITECTURE_AND_SOURCE_INPUTS_READY"


def _gp01_reason(mechanism_id: str, domain: str) -> str:
    reasons = {
        "GP01-L": "COMMITTED_LOCAL_RADIAL_SOURCE_CONTROL_READY",
        "GP01-AQUAL": (
            "SPHERICAL_XCOP_KNOWN_REWRITE_SCORE_ONCE"
            if domain == "XCOP"
            else "FULL_SPARC_3D_SOURCE_AND_BOUNDARY_BLOCKED"
        ),
        "GP01-T1": "ALL_EIGHT_XCOP_CLUSTERS_LACK_UNIQUE_Y100_ANCHOR"
        if domain == "XCOP"
        else "FULL_SPARC_TRANSPORT_SOURCE_BLOCKED",
        "GP01-T2": "ALL_EIGHT_XCOP_CLUSTERS_LACK_UNIQUE_Y100_ANCHOR"
        if domain == "XCOP"
        else "FULL_SPARC_TRANSPORT_SOURCE_BLOCKED",
        "GP01-ELLIPTIC": "SPHERICAL_XCOP_SOURCE_READY_PENDING_EXACT_ADAPTER"
        if domain == "XCOP"
        else "FULL_SPARC_3D_SOURCE_BLOCKED",
        "GP01-TELEGRAPH": "STATIC_PROFILE_HAS_NO_SOURCE_HISTORY_AND_CAUSAL_COMPLETION",
        "GP01-ACTION_PLACEHOLDER": "INCOMPLETE_ACTION_QUARANTINED",
    }
    return reasons[mechanism_id]


def _mechanism_disposition(
    config: Mapping[str, Any], mechanism: Mapping[str, Any], domain: str
) -> tuple[str, str, int | None, int, str]:
    family = mechanism["mechanism_family"]
    if family.startswith("TWELL_"):
        return _twell_disposition(config, mechanism, domain)
    mechanism_id = mechanism["mechanism_id"]
    if family == "GP01":
        row = config["gp01_dispositions"][mechanism_id][domain]
        parameter = row["parameter_cells"]
        declared = row["declared_parameter_cell_count"]
        if mechanism_id == "GP01-ELLIPTIC" and domain == "XCOP":
            if _gate_bound(config, "STATIC_RADIAL_ADAPTER"):
                parameter = "SOURCE_READY_COMPILED_PARAMETER_CELLS_NO_DATA_ELIGIBILITY"
                sealed = int(declared)
            else:
                sealed = 0
        elif parameter == "SOURCE_READY_COMPILED_PARAMETER_CELLS_NO_DATA_ELIGIBILITY":
            sealed = int(declared or 0)
        else:
            sealed = 0
        return row["concept"], parameter, declared, sealed, _gp01_reason(mechanism_id, domain)
    return (
        "THEORY_ONLY",
        "THEORY_ONLY_NO_PARAMETER_CELLS",
        0,
        0,
        "ONTOLOGY_NODE_HAS_NO_EXECUTABLE_SOURCE_CONTRACT",
    )


def iter_matrix_rows(
    config: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> Iterable[dict[str, Any]]:
    observables = predecessor["observable_contracts"]
    stellar_available = set(predecessor["objects"]["XCOP_stellar_profile_available"])
    for mechanism in mechanism_catalog(predecessor):
        for object_id in predecessor["objects"]["SPARC"]:
            observable = observables["SPARC"][0]
            concept, parameter, declared, sealed, reason = _mechanism_disposition(
                config, mechanism, "SPARC"
            )
            yield {
                "mechanism_id": mechanism["mechanism_id"],
                "mechanism_family": mechanism["mechanism_family"],
                "discovery_lane": mechanism["discovery_lane"],
                "object_id": object_id,
                "domain": "SPARC",
                "observable_id": observable["observable_id"],
                "driver_source_status": _combined_driver_status(
                    predecessor, "SPARC", mechanism["drivers"]
                ),
                "concept_readiness": concept,
                "parameter_cell_readiness": parameter,
                "declared_parameter_cell_count": declared,
                "sealed_source_ready_parameter_cell_count": sealed,
                "disposition_reason": reason,
                "observable_source_status": observable["status"],
                "stellar_source_status": "OUT_OF_SCOPE_NOT_OPENED",
                "shared_xray_measurement_ancestry": False,
                "DATA_ELIGIBLE_claimed": False,
                "campaign_authority_granted": False,
                "source_only_no_scoring_authority": True,
            }
        for object_id in predecessor["objects"]["XCOP"]:
            concept, parameter, declared, sealed, reason = _mechanism_disposition(
                config, mechanism, "XCOP"
            )
            for observable in observables["XCOP"]:
                yield {
                    "mechanism_id": mechanism["mechanism_id"],
                    "mechanism_family": mechanism["mechanism_family"],
                    "discovery_lane": mechanism["discovery_lane"],
                    "object_id": object_id,
                    "domain": "XCOP",
                    "observable_id": observable["observable_id"],
                    "driver_source_status": _combined_driver_status(
                        predecessor, "XCOP", mechanism["drivers"]
                    ),
                    "concept_readiness": concept,
                    "parameter_cell_readiness": parameter,
                    "declared_parameter_cell_count": declared,
                    "sealed_source_ready_parameter_cell_count": sealed,
                    "disposition_reason": reason,
                    "observable_source_status": observable["status"],
                    "stellar_source_status": (
                        "SOURCE_AVAILABLE" if object_id in stellar_available else "SOURCE_MISSING"
                    ),
                    "shared_xray_measurement_ancestry": True,
                    "DATA_ELIGIBLE_claimed": False,
                    "campaign_authority_granted": False,
                    "source_only_no_scoring_authority": True,
                }


def matrix_summary(config: Mapping[str, Any], predecessor: Mapping[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    domain_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    parameter_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in iter_matrix_rows(config, predecessor):
        digest.update(_canonical(row))
        digest.update(b"\n")
        count += 1
        domain_counts[row["domain"]] += 1
        concept_counts[row["concept_readiness"]] += 1
        parameter_counts[row["parameter_cell_readiness"]] += 1
        reason_counts[row["disposition_reason"]] += 1
    _require(count == 65100, "streamed matrix count changed")
    return {
        "expanded_tuple_count": count,
        "canonical_row_stream_sha256": digest.hexdigest(),
        "domain_tuple_counts": dict(sorted(domain_counts.items())),
        "concept_readiness_counts": dict(sorted(concept_counts.items())),
        "parameter_cell_readiness_counts": dict(sorted(parameter_counts.items())),
        "disposition_reason_counts": dict(sorted(reason_counts.items())),
        "rows_materialized_in_receipt": 0,
        "normalization": config["matrix_contract"]["normalization"],
    }


def _registry_projection(concept: str) -> tuple[str, bool, str]:
    if concept == "SOURCE_BLOCKED":
        return "SOURCE_BLOCKED", False, "SOURCE_BLOCKED"
    if concept == "QUARANTINED":
        return "QUARANTINED_REVISION_REQUIRED", False, "QUARANTINED"
    if concept == "KNOWN_REWRITE_NONINDEPENDENT":
        return "KNOWN_REWRITE_NONINDEPENDENT", False, "KNOWN_REWRITE_NONINDEPENDENT"
    return "REGISTERED_THEORY_ONLY", False, "THEORY_ONLY"


def _domain_source_ledger_seal(
    config: Mapping[str, Any], predecessor: Mapping[str, Any], registry_domain: str
) -> str:
    if registry_domain == "GALAXIES":
        payload = {
            "objects": config["inherited_sections"]["objects"],
            "ledger": predecessor["object_ledger_seals"],
            "drivers": config["disposition_contract"]["domain_driver_allowlists"]["SPARC"],
            "partition": predecessor["partition_design"],
        }
    elif registry_domain == "CLUSTERS":
        payload = {
            "objects": config["inherited_sections"]["objects"],
            "stellar_available": predecessor["objects"]["XCOP_stellar_profile_available"],
            "stellar_missing": predecessor["objects"]["XCOP_stellar_profile_missing"],
            "drivers": config["disposition_contract"]["domain_driver_allowlists"]["XCOP"],
            "shared_xray_ancestry": True,
            "partition": predecessor["partition_design"],
        }
    elif registry_domain == "GROUPS":
        payload = config["discriminator_source_block"]
    else:
        payload = {
            "domain": "LENSING",
            "status": "SOURCE_BLOCKED",
            "reason": "NO_FROZEN_SOURCE_LEDGER_OR_LIGHT_CLOSURE",
        }
    return content_sha256(payload)


def _manifest_disposition(
    config: Mapping[str, Any], mechanism: Mapping[str, Any], registry_domain: str
) -> tuple[str, str, int | None, int, str]:
    if registry_domain == "GALAXIES":
        return _mechanism_disposition(config, mechanism, "SPARC")
    if registry_domain == "CLUSTERS":
        return _mechanism_disposition(config, mechanism, "XCOP")
    if mechanism["mechanism_family"] == "GRAVITY_LIGHT_ONTOLOGY":
        return (
            "THEORY_ONLY",
            "THEORY_ONLY_NO_PARAMETER_CELLS",
            0,
            0,
            f"{registry_domain}_ONTOLOGY_THEORY_ONLY_NO_SOURCE_LEDGER",
        )
    if mechanism["mechanism_id"] == "GP01-ACTION_PLACEHOLDER":
        return (
            "QUARANTINED",
            "QUARANTINED_NO_PARAMETER_CELL_EXECUTION",
            0,
            0,
            f"{registry_domain}_ACTION_PLACEHOLDER_QUARANTINED",
        )
    return (
        "SOURCE_BLOCKED",
        "SOURCE_BLOCKED_NO_PARAMETER_CELL_EXECUTION",
        None,
        0,
        f"{registry_domain}_NO_FROZEN_SOURCE_LEDGER",
    )


def _source_contract_payload(
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    registry_domain: str,
    concept: str,
    parameter: str,
    reason: str,
    *,
    scope_level: str,
    parameter_cell_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_CONTRACT_SCHEMA,
        "scope_level": scope_level,
        "mechanism_id": mechanism["mechanism_id"],
        "mechanism_family": mechanism["mechanism_family"],
        "parameter_cell_id": parameter_cell_id,
        "registry_domain": registry_domain,
        "concept_readiness": concept,
        "parameter_cell_readiness": parameter,
        "disposition_reason": reason,
        "domain_source_ledger_sha256": _domain_source_ledger_seal(
            config, predecessor, registry_domain
        ),
        "final_bind_gate_sha256": content_sha256(config["final_bind_gates"]),
        "DATA_ELIGIBLE_claimed": False,
        "campaign_authority_granted": False,
        "response_scoring_authorized": False,
    }


def _representability(config: Mapping[str, Any], mechanism: Mapping[str, Any]) -> tuple[bool, str]:
    family = mechanism["mechanism_family"]
    if family.startswith("TWELL_"):
        if _gate_bound(config, "TWELL_SUCCESSOR"):
            return True, "FINAL_TWELL_MECHANISM_CARD_BOUND"
        return False, "PENDING_FINAL_TWELL_MECHANISM_CARD_BIND"
    if family == "GP01":
        return False, "GP01_FINAL_REGISTRY_MECHANISM_CARD_MISSING"
    return False, "ONTOLOGY_NODE_IS_NOT_A_REGISTRY_CANDIDATE_CARD"


def _manifest_row(
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    registry_domain: str,
    *,
    scope_level: str,
    parameter_cell_id: str | None,
) -> dict[str, Any]:
    concept, parameter, declared, sealed, reason = _manifest_disposition(
        config, mechanism, registry_domain
    )
    candidate_status, eligible, execution = _registry_projection(concept)
    contract = _source_contract_payload(
        config,
        predecessor,
        mechanism,
        registry_domain,
        concept,
        parameter,
        reason,
        scope_level=scope_level,
        parameter_cell_id=parameter_cell_id,
    )
    source_contract_sha256 = content_sha256(contract)
    representable, representability_reason = _representability(config, mechanism)
    return {
        "scope_level": scope_level,
        "mechanism_id": mechanism["mechanism_id"],
        "mechanism_family": mechanism["mechanism_family"],
        "discovery_lane": mechanism["discovery_lane"],
        "parameter_cell_id": parameter_cell_id,
        "registry_domain": registry_domain,
        "concept_readiness": concept,
        "parameter_cell_readiness": parameter,
        "declared_parameter_cell_count": declared,
        "sealed_source_ready_parameter_cell_count": sealed,
        "candidate_status": candidate_status,
        "domain_execution": {
            "eligible": eligible,
            "execution_disposition": execution,
            "scored": False,
            "source_contract_sha256": source_contract_sha256,
        },
        "source_contract_sha256": source_contract_sha256,
        "manifest_candidate_representable": representable,
        "representability_reason": representability_reason,
        "disposition_reason": reason,
    }


def iter_manifest_concept_domain_rows(
    config: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> Iterable[dict[str, Any]]:
    for mechanism in mechanism_catalog(predecessor):
        for registry_domain in REGISTRY_DOMAINS:
            yield _manifest_row(
                config,
                predecessor,
                mechanism,
                registry_domain,
                scope_level="CONCEPT",
                parameter_cell_id=None,
            )


def _twell_parameter_cell_ids(config: Mapping[str, Any], mechanism: Mapping[str, Any]) -> list[str]:
    count = int(
        config["disposition_contract"]["architecture_parameter_cell_counts"][
            mechanism["architecture"]
        ]
    )
    if mechanism["mechanism_id"] in {"X19", "X20"}:
        count += 1
    return [f"{mechanism['mechanism_id']}-C{index:03d}" for index in range(1, count + 1)]


def _gp01_elliptic_cell_ids() -> Iterable[str]:
    grid = (
        (1, 2, 4),
        (2.0, 4.0, 8.0),
        (0.1, 1.0, 10.0),
        (0.1, 1.0, 10.0),
        (1, 2),
        (1, 2),
        (0.0, 0.25, 1.0, 4.0),
    )
    for n, a_max, rho, tide, q, power, length in itertools.product(*grid):
        yield (f"GP01E-n{n}-A{a_max:g}-rho{rho:g}-T{tide:g}-q{q}-p{power}-L{length:g}")


def iter_exact_parameter_cells(
    config: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> Iterable[tuple[dict[str, Any], str]]:
    catalog = mechanism_catalog(predecessor)
    by_id = {row["mechanism_id"]: row for row in catalog}
    for mechanism in catalog[:400]:
        for cell_id in _twell_parameter_cell_ids(config, mechanism):
            yield mechanism, cell_id
    for n in (1, 2, 4):
        yield by_id["GP01-L"], f"GP01L-n{n}"
    for n in (1, 2, 4):
        yield by_id["GP01-AQUAL"], f"GP01AQUAL-n{n}"
    for cell_id in _gp01_elliptic_cell_ids():
        yield by_id["GP01-ELLIPTIC"], cell_id


def iter_manifest_parameter_cell_domain_rows(
    config: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> Iterable[dict[str, Any]]:
    for mechanism, cell_id in iter_exact_parameter_cells(config, predecessor):
        for registry_domain in REGISTRY_DOMAINS:
            yield _manifest_row(
                config,
                predecessor,
                mechanism,
                registry_domain,
                scope_level="PARAMETER_CELL",
                parameter_cell_id=cell_id,
            )


def _stream_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    candidate: Counter[str] = Counter()
    execution: Counter[str] = Counter()
    representable: Counter[str] = Counter()
    domain: Counter[str] = Counter()
    unique_source_contracts: set[str] = set()
    for row in rows:
        digest.update(_canonical(row))
        digest.update(b"\n")
        count += 1
        candidate[row["candidate_status"]] += 1
        execution[row["domain_execution"]["execution_disposition"]] += 1
        representable[str(row["manifest_candidate_representable"])] += 1
        domain[row["registry_domain"]] += 1
        unique_source_contracts.add(row["source_contract_sha256"])
    return {
        "slot_count": count,
        "canonical_slot_stream_sha256": digest.hexdigest(),
        "unique_source_contract_sha256_count": len(unique_source_contracts),
        "candidate_status_counts": dict(sorted(candidate.items())),
        "execution_disposition_counts": dict(sorted(execution.items())),
        "manifest_representability_counts": dict(sorted(representable.items())),
        "registry_domain_counts": dict(sorted(domain.items())),
        "rows_materialized_in_receipt": 0,
    }


def manifest_projection_summary(
    config: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> dict[str, Any]:
    concepts = _stream_summary(iter_manifest_concept_domain_rows(config, predecessor))
    cells = _stream_summary(iter_manifest_parameter_cell_domain_rows(config, predecessor))
    _require(concepts["slot_count"] == 1680, "concept manifest projection count changed")
    _require(cells["slot_count"] == 9944, "parameter-cell manifest projection count changed")
    return {
        "source_contract_schema_version": SOURCE_CONTRACT_SCHEMA,
        "concept_domain_slots": concepts,
        "parameter_cell_domain_slots": cells,
        "representability": config["manifest_projection_contract"]["representability"],
        "DATA_ELIGIBLE_claimed": False,
        "campaign_authority_granted": False,
    }


def build_receipt() -> dict[str, Any]:
    config_bytes, module_bytes, test_bytes = _canonical_artifact_bytes()
    config = validate_artifact_integrity(config_bytes, module_bytes, test_bytes)
    predecessor = load_predecessor(config)
    require_final_gates(config)
    catalog = mechanism_catalog(predecessor)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_id": config["contract_id"],
        "status": "SEALED_SOURCE_AVAILABILITY_SUCCESSOR_ZERO_RESPONSE_ACCESS_NO_CAMPAIGN_AUTHORITY",
        "authority_boundary": config["authority_boundary"],
        "claim_boundary": config["claim_boundary"],
        "zero_access": {
            "scientific_response_files_opened_by_generator": 0,
            "scientific_response_rows_opened_by_generator": 0,
            "upstream_receipt_payloads_opened_or_parsed_by_generator": 0,
            "development_response_rows_opened": 0,
            "group_response_rows_opened": 0,
            "lensing_response_rows_opened": 0,
            "confirmation_rows_opened": 0,
            "independent_rows_opened": 0,
            "scientific_scores_computed": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "bindings": {
            "config": {
                "path": _CANONICAL_CONFIG_PATH.as_posix(),
                "sha256": bytes_sha256(config_bytes),
                "semantic_sha256": content_sha256(config),
            },
            "module": {
                "path": _CANONICAL_MODULE_PATH.as_posix(),
                "sha256": bytes_sha256(module_bytes),
                "semantic_sha256": implementation_semantic_sha256(module_bytes),
            },
            "test": {
                "path": _CANONICAL_TEST_PATH.as_posix(),
                "sha256": bytes_sha256(test_bytes),
                "semantic_sha256": text_semantic_sha256(test_bytes),
            },
            "predecessor_counterevidence": config["predecessor_counterevidence"],
            "committed": config["committed_bindings"],
            "final_gates": config["final_bind_gates"],
        },
        "strict_section_sha256": config["section_sha256"],
        "catalog": {
            "mechanisms": len(catalog),
            "TWELL": 400,
            "GP01": 7,
            "ontology_nodes": 13,
            "twell_ordered_ids_sha256": content_sha256(twell_concept_ids()),
            "lane_counts": dict(sorted(Counter(row["discovery_lane"] for row in catalog).items())),
        },
        "objects": {
            "SPARC": len(predecessor["objects"]["SPARC"]),
            "XCOP": len(predecessor["objects"]["XCOP"]),
            "XCOP_stellar_profile_available": predecessor["objects"][
                "XCOP_stellar_profile_available"
            ],
            "XCOP_stellar_profile_missing": predecessor["objects"]["XCOP_stellar_profile_missing"],
            "ledger_seals": predecessor["object_ledger_seals"],
        },
        "observables": predecessor["observable_contracts"],
        "source_rules": {
            "dispositions": config["disposition_contract"],
            "GP01": config["gp01_dispositions"],
            "ontology": config["ontology_disposition"],
            "matched_environment_group_discriminator": config["discriminator_source_block"],
        },
        "matrix": matrix_summary(config, predecessor),
        "manifest_projection": manifest_projection_summary(config, predecessor),
        "partition_design": predecessor["partition_design"],
        "comparators": predecessor["comparator_inventory"],
        "comparator_policy": config["comparator_policy"],
        "legacy_multiplicity_lower_bound": predecessor["legacy_multiplicity_lower_bound"],
        "author_agent_response_exposure_incidents": predecessor["incident_ledger"],
        "campaign_manifest": {
            "status": "NOT_CREATED_BY_THIS_CONTRACT",
            "DATA_ELIGIBLE_claimed": False,
            "response_execution_authorized": False,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise SourceAvailabilityV2Error(f"refusing to overwrite append-only receipt: {path}")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return "CREATED"


def write_receipt() -> str:
    path = _canonical_output_path()
    receipt = build_receipt()
    payload = (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    return _atomic_no_clobber(path, payload)


def validate_receipt_payload(
    stored: Mapping[str, Any],
    *,
    config_bytes: bytes,
    module_bytes: bytes,
    test_bytes: bytes,
) -> None:
    config = validate_artifact_integrity(config_bytes, module_bytes, test_bytes)
    observed = stored.get("content_sha256")
    without_self_hash = {key: value for key, value in stored.items() if key != "content_sha256"}
    _require(observed == content_sha256(without_self_hash), "receipt content hash changed")
    bindings = stored.get("bindings")
    _require(isinstance(bindings, Mapping), "receipt bindings changed")
    expected_bindings = {
        "config": {
            "path": _CANONICAL_CONFIG_PATH.as_posix(),
            "sha256": EXPECTED_CONFIG_FILE_SHA256,
            "semantic_sha256": EXPECTED_CONFIG_CONTENT_SHA256,
        },
        "module": {
            "path": _CANONICAL_MODULE_PATH.as_posix(),
            "sha256": bytes_sha256(module_bytes),
            "semantic_sha256": EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256,
        },
        "test": {
            "path": _CANONICAL_TEST_PATH.as_posix(),
            "sha256": EXPECTED_TEST_FILE_SHA256,
            "semantic_sha256": EXPECTED_TEST_SEMANTIC_SHA256,
        },
    }
    for label, expected in expected_bindings.items():
        _require(bindings.get(label) == expected, f"receipt {label} binding changed")
    _require(content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config seal changed")
    expected = build_receipt()
    _require(dict(stored) == expected, "receipt is not reproducible")


def validate_receipt() -> None:
    path = _canonical_output_path()
    config_bytes, module_bytes, test_bytes = _canonical_artifact_bytes()
    stored = _read_json(path, "v2 zero-response receipt")
    validate_receipt_payload(
        stored,
        config_bytes=config_bytes,
        module_bytes=module_bytes,
        test_bytes=test_bytes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "summary", "manifest-summary"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "validate":
        validate_receipt()
        print("VALID")
    else:
        config, predecessor = load_inputs()
        value = (
            matrix_summary(config, predecessor)
            if args.command == "summary"
            else manifest_projection_summary(config, predecessor)
        )
        print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
