"""Append-only, response-blind foundation for the open-gravity discovery registry.

This module validates governance metadata only.  It has no scientific data adapter,
no score function, and no path to development, confirmation, independent, group, or
lensing response rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_registry_foundation_v1.json")
MECHANISM_SCHEMA_PATH = Path("configs/open_gravity_mechanism_card_v1.schema.json")
RESERVOIR_SCHEMA_PATH = Path("configs/open_gravity_idea_reservoir_v1.schema.json")
CAMPAIGN_SCHEMA_PATH = Path("configs/open_gravity_campaign_manifest_v1.schema.json")
RESERVOIR_PATH = Path("configs/open_gravity_idea_reservoir_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_registry_foundation_v1.py")
TEST_PATH = Path("tests/test_open_gravity_registry_foundation_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-registry-foundation-v1/receipt.json")
OPEN_GOAL_PATH = Path(
    "C:/Users/henry/Documents/Codex/2026-08-21/chec/outputs/"
    "invariant-open-gravity-discovery-next-session-goal-2026-08-30.md"
)
BASELINE_GOAL_PATH = Path(
    "C:/Users/henry/Documents/Codex/2026-08-21/chec/outputs/"
    "invariant-time-well-research-goals-2026-08-30.md"
)

CONFIG_SCHEMA = "invariant-open-gravity-registry-foundation-1.0"
RECEIPT_SCHEMA = "invariant-open-gravity-registry-foundation-receipt-1.0"
REGISTRY_ID = "GRAVITY-LIGHT-GRAMMAR-v1"
REGISTRY_VERSION = "1.0.0"
TWELL_ID = "TWELL-400-v2"
IDEA_RESERVOIR_ID = "IDEA-RESERVOIR-v1"
MULTIPLICITY_LEDGER_ID = "OPEN-GRAVITY-GLOBAL-MULTIPLICITY-v1"
TERMINAL_LEDGER_SCHEMA = "invariant-open-gravity-terminal-campaign-ledger-1.0"
TERMINAL_LEDGER_PATH = Path("runs/gravity/open-gravity-campaign-v1/terminal-ledger.json")
TWELL_IDS_SHA256 = "7388f8982c5014ef6c365d00aa780ba2ecb8b8b3f6786658fb3db36b64c29c5f"
RESERVOIR_GENESIS_SHA256 = "75efef4b47d63dac14b88e02cdb732bf140e4dc9bf088d5e055f00c7d6b81198"
DECISION = "PASS_OPEN_GRAVITY_REGISTRY_FOUNDATION_ZERO_RESPONSE_ACCESS_NO_SCIENTIFIC_SCORE"
EXPECTED_CONFIG_CONTENT_SHA256 = "88a2e3428c1fa1975f8f6182c5cf72bde33a5f544915137233b63141a997b4d0"
EXPECTED_SCHEMA_CONTENT_SHA256 = {
    "mechanism_card": "db05b2540d4caad3d06106466692e1ef74e54695282396ad5316d2078acdf41d",
    "idea_reservoir": "51423f528fe882a0b11f6845a28ec5388357302506fe4360833ad6d0f7e8a436",
    "campaign_manifest": "b7c8ea8959e40249d227795afcdefb994d846b4a6174e691d870490877379879",
}
TRUSTED_SESSION_ID = "OPEN-GRAVITY-DISCOVERY-SESSION-2026-08-30"

SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SCIENTIFIC_STATUSES = (
    "E_ESTABLISHED",
    "C_CONTROLLED_UNOBSERVED_SIGNATURE",
    "H_HYPOTHESIS",
    "A_ANALOGY",
)
IDENTITY_CLASSES = (
    "NEW_CONCEPT",
    "FORMULA_VARIANT",
    "PARAMETER_CELL",
    "IMPLEMENTATION",
    "KNOWN_REWRITE",
    "EMPIRICALLY_DEGENERATE",
)
LANES = ("CORE", "ADJACENT", "ORTHOGONAL", "RIVALS_CONTROLS", "WILDCARD")
DOMAINS = ("GALAXIES", "GROUPS", "CLUSTERS", "LENSING")
CANDIDATE_STATUSES = (
    "REGISTERED_THEORY_ONLY",
    "SOURCE_BLOCKED",
    "READY_FOR_RESPONSE_SCORING",
    "QUARANTINED_REVISION_REQUIRED",
    "KNOWN_REWRITE_NONINDEPENDENT",
)
EXECUTION_DISPOSITIONS = (
    "SEALED_UNOPENED_FOR_SCORING",
    "SOURCE_BLOCKED",
    "THEORY_ONLY",
    "QUARANTINED",
    "KNOWN_REWRITE_NONINDEPENDENT",
    "NOT_APPLICABLE",
)
MULTIPLICITY_DIMENSIONS = (
    "response_scored_campaigns",
    "response_planned_campaigns",
    "adaptive_generations",
    "concepts",
    "registered_candidate_rows",
    "equivalence_families",
    "formula_variants",
    "parameter_cells",
    "hyperparameter_cells",
    "nuisance_scenarios",
    "transformations",
    "object_subsets",
    "observables",
    "metrics",
    "repairs",
    "stopping_decisions",
    "residual_inspired_branches",
    "selection_stages",
    "response_planned_formula_variants",
    "response_planned_domain_executions",
    "response_scored_formula_variants",
    "response_scored_domain_executions",
)
ZERO_ACCESS_FIELDS = (
    "scientific_response_files_opened",
    "scientific_response_rows_opened",
    "development_response_rows_opened",
    "group_response_rows_opened",
    "lensing_response_rows_opened",
    "confirmation_rows_opened",
    "independent_rows_opened",
    "scientific_scores_computed",
    "network_calls",
    "model_calls",
    "paid_calls",
)


class OpenGravityRegistryError(RuntimeError):
    """Raised when a frozen open-gravity governance invariant fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise OpenGravityRegistryError(f"noncanonical governance value: {error}") from error
    return serialized.encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


@dataclass
class MetadataAccessLedger:
    """Records the exact governance files read while rebuilding the receipt."""

    repo: Path
    allowed: Mapping[Path, str]
    opened: dict[str, str] = field(default_factory=dict)

    def read_bytes(self, path: Path) -> bytes:
        resolved = path.resolve()
        normalized = {item.resolve(): kind for item, kind in self.allowed.items()}
        if resolved not in normalized:
            raise OpenGravityRegistryError(f"non-allowlisted read refused: {resolved}")
        display = self.display_path(resolved)
        self.opened[display] = normalized[resolved]
        try:
            return resolved.read_bytes()
        except OSError as error:
            raise OpenGravityRegistryError(
                f"could not read governance metadata: {resolved}"
            ) from error

    def display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo).as_posix()
        except ValueError:
            return path.as_posix()

    def rows(self) -> list[dict[str, str]]:
        return [{"path": path, "artifact_kind": self.opened[path]} for path in sorted(self.opened)]


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON token: {token}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise OpenGravityRegistryError(f"invalid JSON governance metadata: {label}") from error
    if not isinstance(value, dict):
        raise OpenGravityRegistryError(f"JSON object required: {label}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise OpenGravityRegistryError(f"could not read JSON metadata: {path}") from error
    return _load_json_bytes(payload, str(path))


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    config = _load_json(repo / CONFIG_PATH)
    validate_foundation_config(config)
    return config


def _parse_semver(value: str) -> tuple[int, int, int]:
    match = SEMVER_RE.fullmatch(value)
    if match is None:
        raise OpenGravityRegistryError(f"invalid semantic version: {value!r}")
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def twell_concept_ids() -> list[str]:
    atomic = [
        f"TW2-A{architecture:02d}-D{driver:02d}"
        for architecture in range(1, 20)
        for driver in range(1, 21)
    ]
    return atomic + [f"X{compound:02d}" for compound in range(1, 21)]


def _local_ref(root_schema: Mapping[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise OpenGravityRegistryError(f"only local schema references are supported: {reference}")
    value: Any = root_schema
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            raise OpenGravityRegistryError(f"broken local schema reference: {reference}")
        value = value[token]
    return value


def _json_equal(left: Any, right: Any) -> bool:
    return _canonical(left) == _canonical(right)


def _type_matches(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, Mapping)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return (
            isinstance(instance, (int, float))
            and not isinstance(instance, bool)
            and math.isfinite(float(instance))
        )
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    raise OpenGravityRegistryError(f"unsupported schema type in frozen schema: {expected}")


def schema_errors(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Validate the strict JSON-Schema subset used by the frozen governance files."""

    errors: list[str] = []

    def visit(value: Any, rule: Mapping[str, Any], path: str) -> None:
        if "$ref" in rule:
            resolved = _local_ref(schema, str(rule["$ref"]))
            if not isinstance(resolved, Mapping):
                errors.append(f"{path}: reference is not a schema object")
                return
            visit(value, resolved, path)
            return
        if "oneOf" in rule:
            branch_errors: list[list[str]] = []
            for branch in rule["oneOf"]:
                before = len(errors)
                visit(value, branch, path)
                branch_errors.append(errors[before:])
                del errors[before:]
            passing = [row for row in branch_errors if not row]
            if len(passing) != 1:
                errors.append(f"{path}: value does not match exactly one oneOf branch")
            return
        if "const" in rule and not _json_equal(value, rule["const"]):
            errors.append(f"{path}: expected constant {rule['const']!r}")
            return
        if "enum" in rule and not any(_json_equal(value, item) for item in rule["enum"]):
            errors.append(f"{path}: value is outside frozen enum")
            return
        expected_type = rule.get("type")
        if expected_type is not None:
            choices = [expected_type] if isinstance(expected_type, str) else list(expected_type)
            if not any(_type_matches(value, item) for item in choices):
                errors.append(f"{path}: wrong type; expected {choices}")
                return
        if isinstance(value, Mapping):
            required = set(rule.get("required", []))
            missing = sorted(required - set(value))
            if missing:
                errors.append(f"{path}: missing required fields {missing}")
            properties = rule.get("properties", {})
            if rule.get("additionalProperties") is False:
                extra = sorted(set(value) - set(properties))
                if extra:
                    errors.append(f"{path}: undeclared fields {extra}")
            for key, child in properties.items():
                if key in value:
                    visit(value[key], child, f"{path}.{key}")
        if isinstance(value, list):
            minimum = rule.get("minItems")
            if minimum is not None and len(value) < int(minimum):
                errors.append(f"{path}: expected at least {minimum} items")
            if rule.get("uniqueItems") is True:
                encoded = [_canonical(item) for item in value]
                if len(encoded) != len(set(encoded)):
                    errors.append(f"{path}: duplicate array items")
            item_rule = rule.get("items")
            if isinstance(item_rule, Mapping):
                for index, item in enumerate(value):
                    visit(item, item_rule, f"{path}[{index}]")
        if isinstance(value, str):
            minimum = rule.get("minLength")
            if minimum is not None and len(value) < int(minimum):
                errors.append(f"{path}: string is shorter than {minimum}")
            pattern = rule.get("pattern")
            if pattern is not None and re.fullmatch(str(pattern), value) is None:
                errors.append(f"{path}: string does not match {pattern}")
            if rule.get("format") == "date-time":
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    errors.append(f"{path}: invalid date-time")
                else:
                    if parsed.tzinfo is None:
                        errors.append(f"{path}: date-time must include a UTC offset")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                errors.append(f"{path}: number must be finite")
                return
            minimum = rule.get("minimum")
            if minimum is not None and value < minimum:
                errors.append(f"{path}: value is below minimum {minimum}")
            maximum = rule.get("maximum")
            if maximum is not None and value > maximum:
                errors.append(f"{path}: value is above maximum {maximum}")

    visit(instance, schema, "$")
    return errors


def _validate_schema_document(
    schema: Mapping[str, Any], expected_id: str, expected_content_sha256: str
) -> None:
    if (
        schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("$id") != expected_id
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or not isinstance(schema.get("required"), list)
    ):
        raise OpenGravityRegistryError(f"schema contract changed: {expected_id}")
    if content_sha256(schema) != expected_content_sha256:
        raise OpenGravityRegistryError(f"immutable schema semantics changed: {expected_id}")


def validate_foundation_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "registry_id",
        "semantic_version",
        "status",
        "append_only",
        "purpose",
        "source_provenance",
        "scientific_status_labels",
        "scientific_status_policy",
        "identity_rules",
        "version_rules",
        "lifecycle",
        "grammar",
        "twell_400_binding",
        "schema_bindings",
        "campaign_candidate_contract",
        "target_blind_contract",
        "global_multiplicity_contract",
        "one_campaign_terminal_rule",
        "zero_access_accounting",
        "claim_boundary",
        "output_path",
    }
    if set(config) != required:
        raise OpenGravityRegistryError("foundation config fields changed")
    if content_sha256(config) != EXPECTED_CONFIG_CONTENT_SHA256:
        raise OpenGravityRegistryError("immutable foundation config semantics changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["registry_id"] != REGISTRY_ID
        or config["semantic_version"] != REGISTRY_VERSION
        or config["status"] != "FROZEN_FOUNDATION_NO_RESPONSE_ACCESS"
        or config["append_only"] is not True
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise OpenGravityRegistryError("foundation identity or freeze state changed")
    if tuple(config["scientific_status_labels"]) != SCIENTIFIC_STATUSES:
        raise OpenGravityRegistryError("scientific status labels changed")
    if config["scientific_status_policy"] != {
        "card_claim_E_ESTABLISHED_allowlist": [],
        "unallowlisted_E_ESTABLISHED_action": "INCOMPLETE_QUARANTINE",
        "ontology_default_status_is_not_card_claim_authority": True,
    }:
        raise OpenGravityRegistryError("scientific status authority policy changed")
    if tuple(config["identity_rules"]) != IDENTITY_CLASSES:
        raise OpenGravityRegistryError("identity classes changed")
    _parse_semver(str(config["semantic_version"]))
    expected_sources = [
        (
            "INVARIANT-OPEN-GRAVITY-DISCOVERY-GOAL-2026-08-30",
            OPEN_GOAL_PATH.as_posix(),
            "f083db7acb27896b7ede7dec3e415c7ebf5e3211dd1781699985c120e2db3106",
        ),
        (
            "INVARIANT-TIME-WELL-ROADMAP-2026-08-30",
            BASELINE_GOAL_PATH.as_posix(),
            "94305ae9e037200dabc098781a31ee060f0296719621ee262440efb5060f9d79",
        ),
    ]
    observed_sources = [
        (row["document_id"], Path(row["path"]).as_posix(), row["sha256"])
        for row in config["source_provenance"]
    ]
    if observed_sources != expected_sources:
        raise OpenGravityRegistryError("governance source provenance changed")

    grammar = config["grammar"]
    ontology = grammar["ontology_nodes"]
    if [row["id"] for row in ontology] != [f"QG{index:02d}" for index in range(1, 14)]:
        raise OpenGravityRegistryError("gravity-light ontology is not exactly QG01-QG13")
    if any(row["default_status"] not in SCIENTIFIC_STATUSES for row in ontology):
        raise OpenGravityRegistryError("ontology carries an invalid scientific status")
    if [row["id"] for row in grammar["discovery_lanes"]] != list(LANES):
        raise OpenGravityRegistryError("five-lane discovery grammar changed")
    if len(grammar["light_gravity_axes"]) != 13:
        raise OpenGravityRegistryError("light-gravity exploration axes changed")
    expansion = grammar["expansion_protocol"]
    if (
        expansion["between_campaigns_open"] is not True
        or expansion["within_campaign_frozen"] is not True
        or expansion["new_ideas_destination"] != IDEA_RESERVOIR_ID
        or expansion["post_response_ideas_label"] != "ADAPTIVE_DEVELOPMENT"
        or expansion["post_response_ideas_current_campaign_eligible"] is not False
    ):
        raise OpenGravityRegistryError("open-between/frozen-within expansion rule changed")

    binding = config["twell_400_binding"]
    concept_ids = twell_concept_ids()
    if (
        binding["artifact_id"] != TWELL_ID
        or binding["immutable"] is not True
        or len(binding["drivers"]) != 20
        or len(set(binding["drivers"])) != 20
        or len(binding["architectures"]) != 19
        or len(set(binding["architectures"])) != 19
        or binding["compound_ids"] != [f"X{index:02d}" for index in range(1, 21)]
        or binding["atomic_count"] != 380
        or binding["compound_count"] != 20
        or binding["total_count"] != 400
        or len(concept_ids) != 400
        or len(set(concept_ids)) != 400
        or content_sha256(concept_ids) != TWELL_IDS_SHA256
        or binding["ordered_concept_ids_sha256"] != TWELL_IDS_SHA256
        or binding["default_light_closure"] != "L0_NO_LIGHT_CLAIM"
        or binding["default_capture_closure"] != "C0_ISOLATED_CONSERVATIVE"
        or binding["executable_equations_claimed_for_all_400"] is not False
    ):
        raise OpenGravityRegistryError("TWELL-400-v2 immutable binding changed")

    expected_schema_bindings: dict[str, Any] = {
        "mechanism_card": {
            "path": MECHANISM_SCHEMA_PATH.as_posix(),
            "semantic_content_sha256": EXPECTED_SCHEMA_CONTENT_SHA256["mechanism_card"],
        },
        "idea_reservoir": {
            "path": RESERVOIR_SCHEMA_PATH.as_posix(),
            "semantic_content_sha256": EXPECTED_SCHEMA_CONTENT_SHA256["idea_reservoir"],
        },
        "campaign_manifest": {
            "path": CAMPAIGN_SCHEMA_PATH.as_posix(),
            "semantic_content_sha256": EXPECTED_SCHEMA_CONTENT_SHA256["campaign_manifest"],
        },
        "empty_reservoir": RESERVOIR_PATH.as_posix(),
    }
    if config["schema_bindings"] != expected_schema_bindings:
        raise OpenGravityRegistryError("schema path bindings changed")
    candidate_contract = config["campaign_candidate_contract"]
    if (
        tuple(candidate_contract["domains"]) != DOMAINS
        or tuple(candidate_contract["candidate_statuses"]) != CANDIDATE_STATUSES
        or tuple(candidate_contract["execution_dispositions"]) != EXECUTION_DISPOSITIONS
        or len(candidate_contract["rules"]) != 5
        or candidate_contract["postrun_result_receipt_disposition"] != "SCORED"
        or candidate_contract["postrun_scored_claim_forbidden_in_frozen_manifest"] is not True
    ):
        raise OpenGravityRegistryError("campaign candidate disposition contract changed")

    blind = config["target_blind_contract"]
    if (
        blind["anonymous_formula_identifiers"] is not True
        or blind["anonymous_object_identifiers"] is not True
        or blind["target_class_switches_allowed"] is not False
        or blind["object_specific_gravity_tuning_allowed"] is not False
        or blind["score_feedback_to_formula_authors_before_batch_close"] is not False
        or blind["theory_author_may_adjudicate_own_candidate"] is not False
        or "residuals" not in blind["forbidden_generation_inputs"]
        or "confirmation_rows" not in blind["forbidden_generation_inputs"]
    ):
        raise OpenGravityRegistryError("target-blind contract weakened")
    multiplicity = config["global_multiplicity_contract"]
    if (
        multiplicity["ledger_id"] != MULTIPLICITY_LEDGER_ID
        or multiplicity["never_resets"] is not True
        or tuple(multiplicity["count_dimensions"]) != MULTIPLICITY_DIMENSIONS
    ):
        raise OpenGravityRegistryError("global multiplicity contract changed")
    terminal = config["one_campaign_terminal_rule"]
    if (
        terminal["trusted_session_id"] != TRUSTED_SESSION_ID
        or terminal["session_id_is_registry_frozen"] is not True
        or terminal["response_scored_campaign_limit_per_session"] != 1
        or terminal["campaign_ordinal_required"] != 1
        or terminal["automatic_second_campaign_allowed"] is not False
        or terminal["on_adjudication"] != "SESSION_TERMINAL"
        or terminal["zero_survivors_allowed"] is not True
        or terminal["confirmation_slots_must_be_preallocated"] is not True
        or terminal["foundation_campaign_execution_authority"] is not False
        or terminal["manifest_validation_scope"] != "STRUCTURAL_PREAUTHORIZATION_ONLY"
    ):
        raise OpenGravityRegistryError("one-campaign terminal rule changed")
    expected_terminal_ledger = {
        "schema_version": TERMINAL_LEDGER_SCHEMA,
        "path": TERMINAL_LEDGER_PATH.as_posix(),
        "append_only": True,
        "atomic_no_clobber": True,
        "trusted_fields": [
            "session_id",
            "campaign_id",
            "manifest_content_sha256",
            "campaign_ordinal",
            "previous_entry_sha256",
            "adjudication_state",
            "session_terminal",
        ],
        "genesis_requires_no_existing_session_entry": True,
        "second_genesis_forbidden": True,
        "authority_provider": "FUTURE_CAMPAIGN_MANIFEST_PACKAGE",
    }
    if terminal["required_terminal_ledger"] != expected_terminal_ledger:
        raise OpenGravityRegistryError("persisted terminal-ledger contract changed")
    _require_zero_access(config["zero_access_accounting"])
    claims = config["claim_boundary"]
    allowed_true = {
        "registry_foundation_frozen",
        "twell_400_ontology_bound",
        "gravity_light_grammar_frozen",
        "mechanism_card_schema_frozen",
        "idea_reservoir_schema_frozen",
        "campaign_manifest_schema_frozen",
    }
    if any(value is not (key in allowed_true) for key, value in claims.items()):
        raise OpenGravityRegistryError("foundation claim boundary changed")


def _require_zero_access(accounting: Mapping[str, Any]) -> None:
    if set(accounting) != set(ZERO_ACCESS_FIELDS):
        raise OpenGravityRegistryError("zero-access accounting dimensions changed")
    if any(accounting[field] != 0 for field in ZERO_ACCESS_FIELDS):
        raise OpenGravityRegistryError("response or external access is nonzero")


def load_schemas(root: Path | None = None) -> dict[str, dict[str, Any]]:
    repo = _repo_root() if root is None else root.resolve()
    schemas = {
        "mechanism_card": _load_json(repo / MECHANISM_SCHEMA_PATH),
        "idea_reservoir": _load_json(repo / RESERVOIR_SCHEMA_PATH),
        "campaign_manifest": _load_json(repo / CAMPAIGN_SCHEMA_PATH),
    }
    _validate_schema_document(
        schemas["mechanism_card"],
        "urn:invariant:open-gravity:mechanism-card:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["mechanism_card"],
    )
    _validate_schema_document(
        schemas["idea_reservoir"],
        "urn:invariant:open-gravity:idea-reservoir:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["idea_reservoir"],
    )
    _validate_schema_document(
        schemas["campaign_manifest"],
        "urn:invariant:open-gravity:campaign-manifest:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["campaign_manifest"],
    )
    return schemas


FORMULA_PAYLOAD_FIELDS = (
    "source",
    "coupling",
    "action_or_equations",
    "initial_conditions",
    "boundaries",
    "degrees_of_freedom",
    "propagation",
    "state_rule",
    "closures",
    "ledgers",
    "structure",
    "dimensions",
    "parameter_cells",
    "priors",
    "screens",
    "limiting_cases",
)


def mechanism_formula_sha256(card: Mapping[str, Any]) -> str:
    """Derive the formula hash from the complete formula-bearing card payload."""

    try:
        payload = {field_name: card[field_name] for field_name in FORMULA_PAYLOAD_FIELDS}
    except KeyError as error:
        raise OpenGravityRegistryError(f"formula payload is incomplete: {error.args[0]}") from error
    return content_sha256(payload)


def equivalence_fingerprint_sha256(card: Mapping[str, Any]) -> str:
    try:
        return content_sha256(card["equivalence_fingerprint"])
    except KeyError as error:
        raise OpenGravityRegistryError("equivalence fingerprint is missing") from error


def mechanism_card_admission(card: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    _validate_schema_document(
        schema,
        "urn:invariant:open-gravity:mechanism-card:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["mechanism_card"],
    )
    errors = schema_errors(card, schema)
    if not errors:
        stable_id = str(card["stable_concept_id"])
        version = str(card["semantic_version"])
        if card["card_id"] != f"{stable_id}@{version}":
            errors.append("$.card_id: must equal stable_concept_id@semantic_version")
        provenance = card["provenance"]
        if (
            provenance["origin_timing"] == "ADAPTIVE_DEVELOPMENT"
            and not provenance["residual_access_lineage"]
        ):
            errors.append("$.provenance: adaptive ideas require residual access lineage")
        if provenance["origin_timing"] == "PRE_RESPONSE" and provenance["residual_access_lineage"]:
            errors.append("$.provenance: pre-response ideas cannot claim residual lineage")
        version_change = card["version_change"]
        if version_change["kind"] == "INITIAL_REGISTRATION":
            if (
                card["parents"]
                or version_change["previous_card_id"] is not None
                or version_change["previous_card_sha256"] is not None
                or version_change["changed_facets"]
            ):
                errors.append("$.version_change: initial registration cannot name a predecessor")
        elif (
            version_change["previous_card_id"] is None
            or version_change["previous_card_sha256"] is None
            or not version_change["prior_result_retained"]
        ):
            errors.append("$.version_change: a revision must bind and retain its predecessor")
        if stable_id in twell_concept_ids() and version_change["kind"] == "INITIAL_REGISTRATION":
            closures = card["closures"]
            if (
                closures["photon"] != "L0_NO_LIGHT_CLAIM"
                or closures["capture"] != "C0_ISOLATED_CONSERVATIVE"
            ):
                errors.append("$.closures: initial TWELL cards must retain default closures")
        if card["scientific_status"] == "E_ESTABLISHED":
            errors.append(
                "$.scientific_status: E_ESTABLISHED is not self-attestable; the frozen allowlist is empty"
            )
        mechanism = card["action_or_equations"]
        if mechanism["kind"] == "ACTION_PLACEHOLDER" and mechanism["executable"] is not False:
            errors.append("$.action_or_equations: an action placeholder cannot be executable")
        if card["hashes"]["formula_sha256"] != mechanism_formula_sha256(card):
            errors.append("$.hashes.formula_sha256: does not bind the canonical formula payload")
    if errors:
        return {
            "eligible": False,
            "status": "INCOMPLETE_QUARANTINE",
            "errors": sorted(errors),
        }
    if card["action_or_equations"]["kind"] == "ACTION_PLACEHOLDER":
        return {
            "eligible": False,
            "status": "QUARANTINED_REVISION_REQUIRED",
            "errors": ["action placeholder is not an executable candidate"],
        }
    if card["identity_class"] == "KNOWN_REWRITE":
        return {
            "eligible": False,
            "status": "KNOWN_REWRITE_NONINDEPENDENT",
            "errors": [
                "known rewrite shares a prediction family and is not an independent scored leaf"
            ],
        }
    if card["action_or_equations"]["executable"] is not True:
        return {
            "eligible": False,
            "status": "SOURCE_BLOCKED",
            "errors": ["non-executable action or equations cannot enter a response-scored slot"],
        }
    return {"eligible": True, "status": "READY_FOR_THEORY_GATES", "errors": []}


FIELD_FACETS: Mapping[str, set[str]] = {
    "identity_class": {"physical_meaning"},
    "author_agent": {"nonsemantic_provenance_note"},
    "provenance": {"nonsemantic_provenance_note"},
    "lay_mechanism": {"physical_meaning"},
    "novelty_claim": {"physical_meaning"},
    "ontology": {"ontology"},
    "scientific_status": {"physical_meaning"},
    "operational_variables": {"physical_meaning"},
    "source": {"source"},
    "coupling": {"coupling"},
    "action_or_equations": {"equations", "operators"},
    "initial_conditions": {"initial_conditions"},
    "boundaries": {"boundary_realization", "boundary_ontology"},
    "degrees_of_freedom": {"degrees_of_freedom"},
    "propagation": {"propagation_rule"},
    "state_rule": {"state_rule"},
    "closures": {"observable_closure"},
    "ledgers": {"conservation_channel"},
    "structure": {"causal_structure", "physical_meaning"},
    "dimensions": {"equations"},
    "parameter_cells": {"parameter_grid"},
    "priors": {"priors"},
    "screens": {"screens"},
    "limiting_cases": {"limiting_cases"},
    "source_only_data_contract": {"source", "physical_meaning"},
    "synthetic_falsifier": {"synthetic_falsifier"},
    "real_data_discriminator": {"real_data_discriminator"},
    "prior_art": {"nonsemantic_provenance_note"},
    "equivalence_fingerprint": {"equations", "operators", "observable_closure"},
}


def validate_version_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    config: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> str:
    validate_foundation_config(config)
    mechanism_schema = load_schemas()["mechanism_card"] if schema is None else schema
    _validate_schema_document(
        mechanism_schema,
        "urn:invariant:open-gravity:mechanism-card:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["mechanism_card"],
    )
    for label, card in (("previous", previous), ("current", current)):
        errors = schema_errors(card, mechanism_schema)
        if errors:
            raise OpenGravityRegistryError(f"{label} card schema errors: {'; '.join(errors)}")
        if card["card_id"] != f"{card['stable_concept_id']}@{card['semantic_version']}":
            raise OpenGravityRegistryError(f"{label} card identity is inconsistent")
        if card["hashes"]["formula_sha256"] != mechanism_formula_sha256(card):
            raise OpenGravityRegistryError(f"{label} formula hash is not canonical")
    if previous["stable_concept_id"] != current["stable_concept_id"]:
        raise OpenGravityRegistryError("a version transition cannot change stable concept ID")
    old_version = _parse_semver(str(previous["semantic_version"]))
    new_version = _parse_semver(str(current["semantic_version"]))
    change = current["version_change"]
    if change["previous_card_id"] != previous["card_id"]:
        raise OpenGravityRegistryError("version transition does not bind predecessor card ID")
    if change["previous_card_sha256"] != content_sha256(previous):
        raise OpenGravityRegistryError("version transition does not bind predecessor bytes")
    matching_parents = [
        parent
        for parent in current["parents"]
        if parent["card_id"] == previous["card_id"]
        and parent["card_sha256"] == content_sha256(previous)
    ]
    if len(matching_parents) != 1:
        raise OpenGravityRegistryError("predecessor is absent or ambiguous in parent lineage")
    if matching_parents[0]["relation"] not in {"SUPERSEDES", "REPAIRS"}:
        raise OpenGravityRegistryError("predecessor relation is not a revision relation")
    if not any(
        parent["card_id"] == previous["card_id"]
        and parent["card_sha256"] == content_sha256(previous)
        and parent["relation"] in {"SUPERSEDES", "REPAIRS"}
        for parent in current["parents"]
    ):
        raise OpenGravityRegistryError("predecessor is absent from append-only parent lineage")

    declared = set(change["changed_facets"])
    rules = config["version_rules"]
    allowed_facets = set().union(
        rules["major_change_facets"],
        rules["minor_change_facets"],
        rules["patch_change_facets"],
        rules["metadata_only_facets"],
    )
    if not declared or not declared <= allowed_facets:
        raise OpenGravityRegistryError("version transition has missing or unknown changed facets")
    for field_name, facets in FIELD_FACETS.items():
        if not _json_equal(previous[field_name], current[field_name]) and not declared & facets:
            raise OpenGravityRegistryError(
                f"changed field lacks a declared version facet: {field_name}"
            )
    changed_fields = {
        field_name
        for field_name in FIELD_FACETS
        if not _json_equal(previous[field_name], current[field_name])
    }
    formula_payload_changed = any(
        not _json_equal(previous[field_name], current[field_name])
        for field_name in FORMULA_PAYLOAD_FIELDS
    )
    for facet in declared:
        if facet in {"numerical_bug_fix", "spelling", "nonsemantic_provenance_note"}:
            continue
        if not any(facet in FIELD_FACETS[field_name] for field_name in changed_fields):
            raise OpenGravityRegistryError(f"declared version facet has no changed field: {facet}")
    old_hashes = previous["hashes"]
    new_hashes = current["hashes"]
    scientific_facets = set(rules["major_change_facets"]) | set(rules["minor_change_facets"])
    if old_hashes["formula_sha256"] != new_hashes["formula_sha256"] and not (
        declared & scientific_facets
    ):
        raise OpenGravityRegistryError("formula hash changed without a scientific version facet")
    if old_hashes["configuration_sha256"] != new_hashes["configuration_sha256"] and not (
        declared & scientific_facets
    ):
        raise OpenGravityRegistryError(
            "configuration hash changed without a scientific version facet"
        )
    if old_hashes["code_sha256"] != new_hashes["code_sha256"] and not (
        declared & (scientific_facets | {"numerical_bug_fix", "nonsemantic_provenance_note"})
    ):
        raise OpenGravityRegistryError("code hash changed without a version facet")

    major = bool(declared & set(rules["major_change_facets"]))
    minor = bool(declared & set(rules["minor_change_facets"]))
    patch = bool(declared & set(rules["patch_change_facets"]))
    metadata = bool(declared & set(rules["metadata_only_facets"]))
    if major:
        expected_kind = "MAJOR"
        valid_bump = new_version == (old_version[0] + 1, 0, 0)
    elif minor:
        expected_kind = "MINOR"
        valid_bump = (
            new_version[0] == old_version[0]
            and new_version[1] == old_version[1] + 1
            and new_version[2] == 0
        )
    elif patch:
        expected_kind = "PATCH"
        valid_bump = new_version == (old_version[0], old_version[1], old_version[2] + 1)
        if not change["replay_all_affected"]:
            raise OpenGravityRegistryError("numerical bug fix must replay every affected entry")
    elif metadata:
        expected_kind = "METADATA_ONLY"
        valid_bump = new_version == (old_version[0], old_version[1], old_version[2] + 1)
    else:  # pragma: no cover - exhaustive set guard
        raise OpenGravityRegistryError("unclassified version transition")
    if change["kind"] != expected_kind or not valid_bump:
        raise OpenGravityRegistryError(f"version bump does not satisfy {expected_kind}")
    if expected_kind in {"MAJOR", "MINOR"} and formula_payload_changed:
        if old_hashes["formula_sha256"] == new_hashes["formula_sha256"]:
            raise OpenGravityRegistryError(
                "formula-bearing scientific revision must change its canonical formula hash"
            )
        old_symbolic = previous["equivalence_fingerprint"]["canonical_symbolic_sha256"]
        new_symbolic = current["equivalence_fingerprint"]["canonical_symbolic_sha256"]
        if old_symbolic == new_symbolic:
            raise OpenGravityRegistryError(
                "formula-bearing scientific revision must change its canonical-symbolic fingerprint"
            )
    expected_relation = "REPAIRS" if expected_kind == "PATCH" else "SUPERSEDES"
    if matching_parents[0]["relation"] != expected_relation:
        raise OpenGravityRegistryError(f"{expected_kind} requires {expected_relation} lineage")
    protected_hashes = {
        "formula_sha256",
        "configuration_sha256",
        "data_sha256",
        "environment_sha256",
    }
    if expected_kind in {"PATCH", "METADATA_ONLY"} and any(
        old_hashes[key] != new_hashes[key] for key in protected_hashes
    ):
        raise OpenGravityRegistryError(
            f"{expected_kind} may not rebind protected scientific hashes"
        )
    if expected_kind == "METADATA_ONLY" and old_hashes != new_hashes:
        raise OpenGravityRegistryError("METADATA_ONLY may not rebind any implementation hash")
    if expected_kind == "PATCH" and old_hashes["code_sha256"] == new_hashes["code_sha256"]:
        raise OpenGravityRegistryError("PATCH must bind a changed bug-fix implementation")
    if expected_kind == "METADATA_ONLY" and not changed_fields:
        raise OpenGravityRegistryError("METADATA_ONLY must contain an actual metadata change")
    if (
        expected_kind in {"PATCH", "METADATA_ONLY"}
        and previous["equivalence_fingerprint"] != current["equivalence_fingerprint"]
    ):
        raise OpenGravityRegistryError(
            f"{expected_kind} may not rebind the equivalence fingerprint"
        )
    if not change["prior_result_retained"]:
        raise OpenGravityRegistryError("a revision may not erase its predecessor result")
    return expected_kind


def reservoir_entry_sha256(entry: Mapping[str, Any]) -> str:
    payload = dict(entry)
    payload.pop("entry_sha256", None)
    return content_sha256(payload)


def validate_idea_reservoir(
    reservoir: Mapping[str, Any],
    schema: Mapping[str, Any],
    previous: Mapping[str, Any] | None = None,
) -> None:
    _validate_schema_document(
        schema,
        "urn:invariant:open-gravity:idea-reservoir:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["idea_reservoir"],
    )
    errors = schema_errors(reservoir, schema)
    if errors:
        raise OpenGravityRegistryError("idea reservoir schema errors: " + "; ".join(errors))
    if (
        reservoir["schema_version"] != "invariant-open-gravity-idea-reservoir-1.0"
        or reservoir["reservoir_id"] != IDEA_RESERVOIR_ID
        or reservoir["semantic_version"] != "1.0.0"
        or reservoir["append_only"] is not True
        or reservoir["entry_hash_algorithm"] != "sha256-canonical-json-without-entry-sha256"
        or reservoir["genesis_sha256"] != RESERVOIR_GENESIS_SHA256
    ):
        raise OpenGravityRegistryError("idea reservoir immutable header changed")
    entries = reservoir["entries"]
    seen: set[tuple[str, str]] = set()
    latest_by_idea: dict[str, Mapping[str, Any]] = {}
    prior_hash = RESERVOIR_GENESIS_SHA256
    for index, entry in enumerate(entries):
        identity = (entry["idea_id"], entry["entry_version"])
        if identity in seen:
            raise OpenGravityRegistryError("duplicate idea/version in reservoir")
        seen.add(identity)
        if entry["previous_entry_sha256"] != prior_hash:
            raise OpenGravityRegistryError(f"reservoir hash chain breaks at entry {index}")
        if entry["entry_sha256"] != reservoir_entry_sha256(entry):
            raise OpenGravityRegistryError(f"reservoir entry hash changed at entry {index}")
        if entry["origin_timing"] == "ADAPTIVE_DEVELOPMENT":
            if not entry["residual_source_campaigns"]:
                raise OpenGravityRegistryError("adaptive reservoir entry lacks residual lineage")
        elif entry["residual_source_campaigns"]:
            raise OpenGravityRegistryError("pre-response reservoir entry claims residual lineage")
        if entry["current_campaign_scoring_allowed"] is not False:
            raise OpenGravityRegistryError("reservoir entry leaked into the current campaign")
        same_idea_previous = latest_by_idea.get(str(entry["idea_id"]))
        if same_idea_previous is None:
            if entry["previous_same_idea_entry_sha256"] is not None:
                raise OpenGravityRegistryError(
                    "first same-idea version binds a nonexistent predecessor"
                )
        else:
            if _parse_semver(str(entry["entry_version"])) <= _parse_semver(
                str(same_idea_previous["entry_version"])
            ):
                raise OpenGravityRegistryError("same-idea reservoir version is not monotonic")
            if entry["previous_same_idea_entry_sha256"] != same_idea_previous["entry_sha256"]:
                raise OpenGravityRegistryError("same-idea reservoir lineage is not append-only")
        latest_by_idea[str(entry["idea_id"])] = entry
        prior_hash = entry["entry_sha256"]
    if previous is not None:
        validate_idea_reservoir(previous, schema)
        old_entries = previous["entries"]
        if len(entries) < len(old_entries) or entries[: len(old_entries)] != old_entries:
            raise OpenGravityRegistryError("idea reservoir is not append-only")


def append_reservoir_entry(
    reservoir: Mapping[str, Any], entry_without_hashes: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, Any]:
    validate_idea_reservoir(reservoir, schema)
    result = json.loads(json.dumps(reservoir))
    entries = result["entries"]
    entry = dict(entry_without_hashes)
    entry["previous_entry_sha256"] = (
        entries[-1]["entry_sha256"] if entries else RESERVOIR_GENESIS_SHA256
    )
    same_idea_entries = [row for row in entries if row["idea_id"] == entry.get("idea_id")]
    entry["previous_same_idea_entry_sha256"] = (
        same_idea_entries[-1]["entry_sha256"] if same_idea_entries else None
    )
    entry["entry_sha256"] = reservoir_entry_sha256(entry)
    entries.append(entry)
    validate_idea_reservoir(result, schema, previous=reservoir)
    return result


def campaign_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_content_sha256", None)
    return content_sha256(payload)


def _unique(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> None:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise OpenGravityRegistryError(f"duplicate {label}")


def partition_object_ledger_sha256(object_ids: Sequence[str]) -> str:
    identifiers = [str(value) for value in object_ids]
    if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
        raise OpenGravityRegistryError("partition object IDs must be unique and sorted")
    return content_sha256(identifiers)


def mechanism_card_set_sha256(cards: Sequence[Mapping[str, Any]]) -> str:
    rows = sorted(
        ({"card_id": str(card["card_id"]), "card_sha256": content_sha256(card)} for card in cards),
        key=lambda row: row["card_id"],
    )
    if len(rows) != len({row["card_id"] for row in rows}):
        raise OpenGravityRegistryError("duplicate live mechanism card ID")
    return content_sha256(rows)


def campaign_equivalence_ledger_sha256(candidates: Sequence[Mapping[str, Any]]) -> str:
    rows = sorted(
        (
            {
                "candidate_id": row["candidate_id"],
                "card_sha256": row["card_sha256"],
                "formula_sha256": row["formula_sha256"],
                "equivalence_family_id": row["equivalence_family_id"],
                "equivalence_fingerprint_sha256": row["equivalence_fingerprint_sha256"],
            }
            for row in candidates
        ),
        key=lambda row: (row["candidate_id"], row["card_sha256"]),
    )
    return content_sha256(rows)


def trusted_session_contract_sha256(config: Mapping[str, Any]) -> str:
    return content_sha256(config["one_campaign_terminal_rule"])


def terminal_ledger_contract_sha256(config: Mapping[str, Any]) -> str:
    return content_sha256(config["one_campaign_terminal_rule"]["required_terminal_ledger"])


def validate_campaign_manifest(
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    previous_manifest: Mapping[str, Any] | None = None,
    *,
    mechanism_cards: Sequence[Mapping[str, Any]] | None = None,
    root: Path | None = None,
) -> None:
    """Validate a structurally frozen, unrun campaign without granting execution authority."""

    validate_foundation_config(config)
    _validate_schema_document(
        schema,
        "urn:invariant:open-gravity:campaign-manifest:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["campaign_manifest"],
    )
    errors = schema_errors(manifest, schema)
    if errors:
        raise OpenGravityRegistryError("campaign manifest schema errors: " + "; ".join(errors))
    if manifest["manifest_state"] != "FROZEN_UNRUN":
        raise OpenGravityRegistryError(
            "a frozen manifest is immutable; execution state belongs in receipts"
        )
    if manifest["manifest_content_sha256"] != campaign_manifest_sha256(manifest):
        raise OpenGravityRegistryError("campaign manifest content hash changed")

    repo = _repo_root() if root is None else root.resolve()
    live_receipt = build_receipt(repo)
    cards = list(mechanism_cards or [])
    if not cards:
        raise OpenGravityRegistryError("campaign requires its complete live mechanism-card set")
    card_rows: dict[str, Mapping[str, Any]] = {}
    card_admissions: dict[str, dict[str, Any]] = {}
    mechanism_schema = load_schemas(repo)["mechanism_card"]
    for card in cards:
        card_id = str(card.get("card_id"))
        if card_id in card_rows:
            raise OpenGravityRegistryError("duplicate live mechanism card ID")
        card_rows[card_id] = card
        card_admissions[card_id] = mechanism_card_admission(card, mechanism_schema)

    candidates = manifest["candidate_versions"]
    _unique(candidates, "anonymous_formula_id", "anonymous formula ID")
    _unique(candidates, "candidate_id", "candidate ID")
    _unique(candidates, "card_id", "candidate card ID")
    if set(card_rows) != {str(row["card_id"]) for row in candidates}:
        raise OpenGravityRegistryError("live card set differs from the frozen candidate card set")
    if {row["lane"] for row in candidates} != set(LANES):
        raise OpenGravityRegistryError(
            "every frozen campaign must populate all five discovery lanes"
        )
    lane_counts = Counter(row["lane"] for row in candidates)
    for lane in LANES:
        if lane_counts[lane] > manifest["budgets"]["lane_candidate_limits"][lane]:
            raise OpenGravityRegistryError(f"candidate count exceeds frozen lane budget: {lane}")

    formula_families: dict[str, set[str]] = {}
    planned_family_members: Counter[str] = Counter()
    planned_formula_hashes: set[str] = set()
    planned_domain_count = 0
    for candidate in candidates:
        card = card_rows[str(candidate["card_id"])]
        admission = card_admissions[str(candidate["card_id"])]
        exact_card_fields = {
            "candidate_id": card["stable_concept_id"],
            "card_id": card["card_id"],
            "semantic_version": card["semantic_version"],
            "scientific_status": card["scientific_status"],
            "identity_class": card["identity_class"],
            "mechanism_kind": card["action_or_equations"]["kind"],
            "mechanism_executable": card["action_or_equations"]["executable"],
            "card_sha256": content_sha256(card),
            "formula_sha256": card["hashes"]["formula_sha256"],
            "configuration_sha256": card["hashes"]["configuration_sha256"],
            "equivalence_fingerprint_sha256": equivalence_fingerprint_sha256(card),
        }
        for field_name, expected in exact_card_fields.items():
            if candidate[field_name] != expected:
                raise OpenGravityRegistryError(
                    f"candidate does not bind live card field {field_name}: {candidate['candidate_id']}"
                )
        if candidate["scientific_status"] == "E_ESTABLISHED":
            raise OpenGravityRegistryError(
                "an unallowlisted card may not self-assert E_ESTABLISHED"
            )
        if admission["status"] == "INCOMPLETE_QUARANTINE":
            raise OpenGravityRegistryError(
                f"incomplete live card cannot enter a frozen campaign: {candidate['candidate_id']}"
            )
        allowed_statuses = {
            "READY_FOR_THEORY_GATES": {
                "REGISTERED_THEORY_ONLY",
                "READY_FOR_RESPONSE_SCORING",
            },
            "SOURCE_BLOCKED": {"SOURCE_BLOCKED"},
            "QUARANTINED_REVISION_REQUIRED": {"QUARANTINED_REVISION_REQUIRED"},
            "KNOWN_REWRITE_NONINDEPENDENT": {"KNOWN_REWRITE_NONINDEPENDENT"},
        }
        if candidate["candidate_status"] not in allowed_statuses.get(admission["status"], set()):
            raise OpenGravityRegistryError(
                f"candidate status overclaims live-card admission: {candidate['candidate_id']}"
            )

        domain_execution = candidate["domain_execution"]
        if set(domain_execution) != set(DOMAINS):
            raise OpenGravityRegistryError("candidate domain execution keys changed")
        dispositions = [domain_execution[domain]["execution_disposition"] for domain in DOMAINS]
        for domain in DOMAINS:
            execution = domain_execution[domain]
            is_planned = execution["execution_disposition"] == "SEALED_UNOPENED_FOR_SCORING"
            if execution["scored"] is not False:
                raise OpenGravityRegistryError(
                    f"a frozen-unrun manifest cannot claim a scored domain: {candidate['candidate_id']}:{domain}"
                )
            if execution["eligible"] is not is_planned:
                raise OpenGravityRegistryError(
                    f"domain eligibility flag disagrees with sealed disposition: {candidate['candidate_id']}:{domain}"
                )
            if is_planned:
                planned_domain_count += 1

        status = candidate["candidate_status"]
        permitted_dispositions = {
            "REGISTERED_THEORY_ONLY": {"THEORY_ONLY", "NOT_APPLICABLE"},
            "SOURCE_BLOCKED": {"SOURCE_BLOCKED", "NOT_APPLICABLE"},
            "READY_FOR_RESPONSE_SCORING": {
                "SEALED_UNOPENED_FOR_SCORING",
                "SOURCE_BLOCKED",
                "NOT_APPLICABLE",
            },
            "QUARANTINED_REVISION_REQUIRED": {"QUARANTINED", "NOT_APPLICABLE"},
            "KNOWN_REWRITE_NONINDEPENDENT": {
                "KNOWN_REWRITE_NONINDEPENDENT",
                "NOT_APPLICABLE",
            },
        }[status]
        if not set(dispositions) <= permitted_dispositions:
            raise OpenGravityRegistryError(
                f"candidate has a disposition incompatible with its status: {candidate['candidate_id']}"
            )
        required_disposition = {
            "REGISTERED_THEORY_ONLY": "THEORY_ONLY",
            "SOURCE_BLOCKED": "SOURCE_BLOCKED",
            "READY_FOR_RESPONSE_SCORING": "SEALED_UNOPENED_FOR_SCORING",
            "QUARANTINED_REVISION_REQUIRED": "QUARANTINED",
            "KNOWN_REWRITE_NONINDEPENDENT": "KNOWN_REWRITE_NONINDEPENDENT",
        }[status]
        if required_disposition not in dispositions:
            raise OpenGravityRegistryError(
                f"candidate status has no matching domain disposition: {candidate['candidate_id']}"
            )
        candidate_planned = "SEALED_UNOPENED_FOR_SCORING" in dispositions
        if candidate_planned:
            if (
                admission["eligible"] is not True
                or candidate["mechanism_executable"] is not True
                or candidate["mechanism_kind"] == "ACTION_PLACEHOLDER"
                or candidate["identity_class"] == "KNOWN_REWRITE"
            ):
                raise OpenGravityRegistryError(
                    f"non-executable, placeholder, or rewrite candidate entered a planned scoring slot: {candidate['candidate_id']}"
                )
            planned_formula_hashes.add(str(candidate["formula_sha256"]))
            planned_family_members[str(candidate["equivalence_family_id"])] += 1

        formula_families.setdefault(str(candidate["formula_sha256"]), set()).add(
            str(candidate["equivalence_family_id"])
        )
    split_formulas = sorted(
        formula_sha for formula_sha, families in formula_families.items() if len(families) != 1
    )
    if split_formulas:
        raise OpenGravityRegistryError(
            "identical executable formula hashes were split across equivalence families"
        )
    if any(count > 1 for count in planned_family_members.values()):
        raise OpenGravityRegistryError(
            "more than one member of an equivalence family entered planned scoring slots"
        )

    binding = manifest["registry_binding"]
    expected_binding = {
        "registry_id": REGISTRY_ID,
        "semantic_version": REGISTRY_VERSION,
        "foundation_receipt_sha256": live_receipt["content_sha256"],
        "mechanism_card_set_sha256": mechanism_card_set_sha256(cards),
        "equivalence_ledger_sha256": campaign_equivalence_ledger_sha256(candidates),
        "trusted_session_contract_sha256": trusted_session_contract_sha256(config),
        "twell_400_ids_sha256": TWELL_IDS_SHA256,
    }
    if binding != expected_binding:
        raise OpenGravityRegistryError(
            "campaign does not bind the live registry, cards, and equivalence ledger"
        )

    for key, label in (
        ("parameter_cells", "parameter cell"),
        ("hyperparameter_cells", "hyperparameter cell"),
        ("nuisance_cases", "nuisance case"),
    ):
        _unique(manifest[key], "cell_id", label)
    for key, label in (
        ("transformations", "transformation"),
        ("object_subsets", "object subset"),
        ("observables", "observable"),
        ("metrics", "metric"),
        ("comparators", "comparator"),
        ("repairs", "repair"),
        ("stopping_decisions", "stopping decision"),
        ("selection_stages", "selection stage"),
    ):
        _unique(manifest[key], "item_id", label)

    data = manifest["data_roles_and_splits"]
    partitions = data["source_partitions"] + data["response_partitions"]
    _unique(partitions, "partition_id", "data partition")
    if any(row["role"] != "SOURCE_ONLY" for row in data["source_partitions"]):
        raise OpenGravityRegistryError("source partitions may contain only source-side data")
    response_roles = [row["role"] for row in data["response_partitions"]]
    required_roles = (
        "DEVELOPMENT_PILOT",
        "DEVELOPMENT_FULL",
        "CONFIRMATION_SEALED",
        "INDEPENDENT_SEALED",
    )
    if sorted(response_roles) != sorted(required_roles):
        raise OpenGravityRegistryError(
            "campaign requires exactly one partition for each response role"
        )
    for partition in partitions:
        expected_ledger_sha = partition_object_ledger_sha256(partition["anonymous_object_ids"])
        if partition["object_ledger_sha256"] != expected_ledger_sha:
            raise OpenGravityRegistryError(
                f"partition membership ledger hash changed: {partition['partition_id']}"
            )
    by_role = {row["role"]: row for row in data["response_partitions"]}
    pilot = set(by_role["DEVELOPMENT_PILOT"]["anonymous_object_ids"])
    full = set(by_role["DEVELOPMENT_FULL"]["anonymous_object_ids"])
    if data["pilot_full_relation"] == "DISJOINT":
        if pilot & full or pilot == full:
            raise OpenGravityRegistryError("pilot/full ledgers are not disjoint")
    elif not pilot < full:
        raise OpenGravityRegistryError("pilot must be a proper subset of full")
    confirmation_ids = set(by_role["CONFIRMATION_SEALED"]["anonymous_object_ids"])
    independent_ids = set(by_role["INDEPENDENT_SEALED"]["anonymous_object_ids"])
    development_ids = pilot | full
    if (
        confirmation_ids & development_ids
        or independent_ids & development_ids
        or confirmation_ids & independent_ids
    ):
        raise OpenGravityRegistryError(
            "development, confirmation, and independent membership ledgers overlap"
        )
    if data["confirmation_forbidden_in_campaign"] is not True:
        raise OpenGravityRegistryError("confirmation cannot be opened by this development campaign")

    expected_current = {
        "response_scored_campaigns": 0,
        "response_planned_campaigns": 1,
        "adaptive_generations": len(manifest["adaptive_generation_ids"]),
        "concepts": len({row["candidate_id"] for row in candidates}),
        "registered_candidate_rows": len(candidates),
        "equivalence_families": len({row["equivalence_family_id"] for row in candidates}),
        "formula_variants": len(formula_families),
        "parameter_cells": len(manifest["parameter_cells"]),
        "hyperparameter_cells": len(manifest["hyperparameter_cells"]),
        "nuisance_scenarios": len(manifest["nuisance_cases"]),
        "transformations": len(manifest["transformations"]),
        "object_subsets": len(manifest["object_subsets"]),
        "observables": len(manifest["observables"]),
        "metrics": len(manifest["metrics"]),
        "repairs": len(manifest["repairs"]),
        "stopping_decisions": len(manifest["stopping_decisions"]),
        "residual_inspired_branches": len(manifest["residual_inspired_branch_ids"]),
        "selection_stages": len(manifest["selection_stages"]),
        "response_planned_formula_variants": len(planned_formula_hashes),
        "response_planned_domain_executions": planned_domain_count,
        "response_scored_formula_variants": 0,
        "response_scored_domain_executions": 0,
    }
    ledger = manifest["global_multiplicity_ledger"]
    if ledger["ledger_id"] != MULTIPLICITY_LEDGER_ID or ledger["never_resets"] is not True:
        raise OpenGravityRegistryError("global multiplicity ledger identity changed")
    before = ledger["counts_before"]
    current = ledger["counts_this_campaign"]
    after = ledger["counts_after"]
    if set(before) != set(MULTIPLICITY_DIMENSIONS):
        raise OpenGravityRegistryError("global multiplicity dimensions changed")
    if current != expected_current:
        raise OpenGravityRegistryError("campaign multiplicity counts are not exactly derived")
    for dimension in MULTIPLICITY_DIMENSIONS:
        if after[dimension] != before[dimension] + current[dimension]:
            raise OpenGravityRegistryError(
                f"global multiplicity reset or arithmetic error: {dimension}"
            )
    if (
        ledger["campaign_sequence"] != 1
        or ledger["previous_manifest_sha256"] != "GENESIS"
        or any(before.values())
        or previous_manifest is not None
    ):
        raise OpenGravityRegistryError(
            "this trusted v1 session authorizes only the exact genesis response campaign"
        )

    thresholds = manifest["promotion_thresholds"]
    numeric_thresholds = (
        thresholds["minimum_meaningful_improvement"],
        thresholds["selection_adjusted_evidence_threshold"],
        thresholds["leave_one_object_out_minimum"],
    )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in numeric_thresholds
    ):
        raise OpenGravityRegistryError("promotion thresholds must be finite")
    if not (
        float(thresholds["minimum_meaningful_improvement"]) >= 0.0
        and 0.0 <= float(thresholds["selection_adjusted_evidence_threshold"]) <= 1.0
    ):
        raise OpenGravityRegistryError("promotion thresholds are outside frozen bounds")
    if any(
        not math.isfinite(float(row["maximum"]))
        for row in manifest["worst_case_and_subgroup_ceilings"]
    ):
        raise OpenGravityRegistryError("worst-case ceilings must be finite")

    confirmation = manifest["confirmation"]
    slots = confirmation["slots"]
    if len(slots) != confirmation["K"]:
        raise OpenGravityRegistryError("confirmation slot count does not equal frozen K")
    _unique(slots, "slot_id", "confirmation slot")
    if (
        sum(row["evidence_budget_units"] for row in slots)
        != confirmation["evidence_budget_units_total"]
    ):
        raise OpenGravityRegistryError("confirmation evidence budget is not exactly preallocated")
    if any(not row["sealed"] or row["opened"] or row["candidate_id"] is not None for row in slots):
        raise OpenGravityRegistryError("confirmation slots must remain sealed at manifest freeze")

    blind = manifest["target_blind_contract"]
    frozen_blind = config["target_blind_contract"]
    for key in (
        "anonymous_formula_identifiers",
        "anonymous_object_identifiers",
        "target_class_switches_allowed",
        "object_specific_gravity_tuning_allowed",
        "score_feedback_to_formula_authors_before_batch_close",
        "theory_author_may_adjudicate_own_candidate",
        "legitimate_source_metadata_visible",
        "forbidden_generation_inputs",
    ):
        if blind[key] != frozen_blind[key]:
            raise OpenGravityRegistryError(f"campaign weakens target-blind field: {key}")

    terminal = manifest["session_terminal_contract"]
    if (
        terminal["session_id"] != TRUSTED_SESSION_ID
        or terminal["trusted_session_contract_sha256"] != trusted_session_contract_sha256(config)
        or terminal["terminal_ledger_contract_sha256"] != terminal_ledger_contract_sha256(config)
        or terminal["campaign_execution_authority"] != "WITHHELD_PENDING_PERSISTED_TERMINAL_LEDGER"
        or terminal["response_scored_campaign_limit"] != 1
        or terminal["response_scored_campaign_ordinal"] != 1
        or terminal["automatic_second_campaign_allowed"] is not False
        or terminal["on_adjudication"] != "SESSION_TERMINAL"
        or terminal["post_freeze_new_or_repaired_idea_destination"] != IDEA_RESERVOIR_ID
        or terminal["zero_survivors_allowed"] is not True
    ):
        raise OpenGravityRegistryError("one-campaign trusted-session rule is not exact")
    _require_zero_access(manifest["zero_access_at_freeze"])


def validate_session_campaign_set(
    manifests: Sequence[Mapping[str, Any]], config: Mapping[str, Any] | None = None
) -> None:
    """Enforce the terminal one-response-scored-campaign rule across a session."""

    frozen_config = load_config() if config is None else config
    validate_foundation_config(frozen_config)
    response_scored_count = 0
    campaign_ids: set[str] = set()
    for manifest in manifests:
        campaign_id = str(manifest["campaign_id"])
        if campaign_id in campaign_ids:
            raise OpenGravityRegistryError("duplicate campaign manifest in session ledger")
        campaign_ids.add(campaign_id)
        terminal = manifest["session_terminal_contract"]
        if terminal["session_id"] != TRUSTED_SESSION_ID or terminal[
            "trusted_session_contract_sha256"
        ] != trusted_session_contract_sha256(frozen_config):
            raise OpenGravityRegistryError("manifest relabeled the trusted session")
        if (
            terminal["terminal_ledger_contract_sha256"]
            != terminal_ledger_contract_sha256(frozen_config)
            or terminal["campaign_execution_authority"]
            != "WITHHELD_PENDING_PERSISTED_TERMINAL_LEDGER"
        ):
            raise OpenGravityRegistryError("manifest lacks the frozen terminal-ledger contract")
        if manifest["response_scored_campaign"]:
            response_scored_count += 1
        if (
            terminal["response_scored_campaign_limit"] != 1
            or terminal["response_scored_campaign_ordinal"] != 1
            or terminal["automatic_second_campaign_allowed"] is not False
            or terminal["on_adjudication"] != "SESSION_TERMINAL"
        ):
            raise OpenGravityRegistryError("session ledger weakens the one-campaign terminal rule")
    if response_scored_count > 1:
        raise OpenGravityRegistryError("more than one response-scored campaign in trusted session")


def assert_campaign_execution_authority(
    manifest: Mapping[str, Any], terminal_ledger_receipt: Mapping[str, Any] | None = None
) -> None:
    """Fail closed until a separate package implements the persisted terminal ledger."""

    del manifest, terminal_ledger_receipt
    raise OpenGravityRegistryError(
        "campaign execution authority is withheld until the future manifest package "
        "implements the frozen append-only terminal-ledger contract"
    )


def _governance_allowlist(repo: Path) -> dict[Path, str]:
    allowed: dict[Path, str] = {
        repo / CONFIG_PATH: "registry_config",
        repo / MECHANISM_SCHEMA_PATH: "mechanism_card_schema",
        repo / RESERVOIR_SCHEMA_PATH: "idea_reservoir_schema",
        repo / CAMPAIGN_SCHEMA_PATH: "campaign_manifest_schema",
        repo / RESERVOIR_PATH: "empty_idea_reservoir",
        repo / MODULE_PATH: "validator_implementation",
        repo / TEST_PATH: "validator_tests",
        OPEN_GOAL_PATH: "governance_source_document",
        BASELINE_GOAL_PATH: "governance_source_document",
    }
    return allowed


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    ledger = MetadataAccessLedger(repo, _governance_allowlist(repo))
    config_bytes = ledger.read_bytes(repo / CONFIG_PATH)
    config = _load_json_bytes(config_bytes, CONFIG_PATH.as_posix())
    validate_foundation_config(config)

    schema_paths = {
        "mechanism_card": MECHANISM_SCHEMA_PATH,
        "idea_reservoir": RESERVOIR_SCHEMA_PATH,
        "campaign_manifest": CAMPAIGN_SCHEMA_PATH,
    }
    schemas: dict[str, dict[str, Any]] = {}
    schema_bytes: dict[str, bytes] = {}
    for schema_id, path in schema_paths.items():
        schema_bytes[schema_id] = ledger.read_bytes(repo / path)
        schemas[schema_id] = _load_json_bytes(schema_bytes[schema_id], path.as_posix())
    _validate_schema_document(
        schemas["mechanism_card"],
        "urn:invariant:open-gravity:mechanism-card:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["mechanism_card"],
    )
    _validate_schema_document(
        schemas["idea_reservoir"],
        "urn:invariant:open-gravity:idea-reservoir:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["idea_reservoir"],
    )
    _validate_schema_document(
        schemas["campaign_manifest"],
        "urn:invariant:open-gravity:campaign-manifest:1.0",
        EXPECTED_SCHEMA_CONTENT_SHA256["campaign_manifest"],
    )
    reservoir_bytes = ledger.read_bytes(repo / RESERVOIR_PATH)
    reservoir = _load_json_bytes(reservoir_bytes, RESERVOIR_PATH.as_posix())
    validate_idea_reservoir(reservoir, schemas["idea_reservoir"])

    provenance_receipts: list[dict[str, Any]] = []
    for row in config["source_provenance"]:
        path = Path(row["path"])
        observed = _sha256_bytes(ledger.read_bytes(path))
        if observed != row["sha256"]:
            raise OpenGravityRegistryError(f"governance source document changed: {path}")
        provenance_receipts.append(
            {"document_id": row["document_id"], "path": path.as_posix(), "sha256": observed}
        )

    implementation_sha = _sha256_bytes(ledger.read_bytes(repo / MODULE_PATH))
    test_sha = _sha256_bytes(ledger.read_bytes(repo / TEST_PATH))
    schema_receipts = {
        schema_id: {
            "path": path.as_posix(),
            "file_sha256": _sha256_bytes(schema_bytes[schema_id]),
            "semantic_content_sha256": content_sha256(schemas[schema_id]),
            "schema_id": schemas[schema_id]["$id"],
            "strict_top_level": schemas[schema_id]["additionalProperties"] is False,
        }
        for schema_id, path in schema_paths.items()
    }
    zero_access = dict(config["zero_access_accounting"])
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "registry_id": REGISTRY_ID,
        "semantic_version": REGISTRY_VERSION,
        "status": "MUTATION_FROZEN_FOUNDATION_ONLY",
        "decision": DECISION,
        "bindings": {
            "config_path": CONFIG_PATH.as_posix(),
            "config_file_sha256": _sha256_bytes(config_bytes),
            "config_content_sha256": content_sha256(config),
            "implementation_path": MODULE_PATH.as_posix(),
            "implementation_file_sha256": implementation_sha,
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": test_sha,
            "schemas": schema_receipts,
            "idea_reservoir_path": RESERVOIR_PATH.as_posix(),
            "idea_reservoir_file_sha256": _sha256_bytes(reservoir_bytes),
            "governance_sources": provenance_receipts,
        },
        "registry": {
            "ontology_nodes": 13,
            "light_gravity_axes": 13,
            "discovery_lanes": list(LANES),
            "scientific_status_labels": list(SCIENTIFIC_STATUSES),
            "identity_classes": list(IDENTITY_CLASSES),
            "candidate_statuses": list(CANDIDATE_STATUSES),
            "execution_domains": list(DOMAINS),
            "execution_dispositions": list(EXECUTION_DISPOSITIONS),
            "postrun_result_receipt_disposition": "SCORED",
            "open_between_campaigns": True,
            "frozen_within_campaigns": True,
        },
        "twell_400_binding": {
            "artifact_id": TWELL_ID,
            "atomic_concepts": 380,
            "compound_concepts": 20,
            "total_concepts": 400,
            "ordered_concept_ids_sha256": content_sha256(twell_concept_ids()),
            "immutable_ontology": True,
            "all_400_executable_claimed": False,
        },
        "enforced_controls": {
            "missing_card_fields_quarantine": True,
            "unallowlisted_established_claims_quarantine": True,
            "nonexecutable_and_placeholder_candidates_never_scored": True,
            "known_rewrites_never_independently_scored": True,
            "semantic_identity_and_version_rules": True,
            "patch_and_metadata_protected_hashes": True,
            "schema_semantics_content_pinned": True,
            "append_only_idea_hash_chain": True,
            "same_idea_semver_monotonic": True,
            "adaptive_ideas_future_campaign_only": True,
            "campaign_schema_validator_available": True,
            "live_receipt_card_and_equivalence_binding": True,
            "identical_formula_one_equivalence_family": True,
            "pilot_full_membership_relation_computed": True,
            "global_multiplicity_non_reset": True,
            "all_multiplicity_dimensions_exactly_derived": True,
            "registered_and_scored_variants_counted_separately": True,
            "one_response_scored_campaign_per_session": True,
            "trusted_session_id_registry_frozen": True,
            "campaign_execution_authority_withheld_until_terminal_ledger": True,
            "frozen_unrun_manifest_scored_claims_forbidden": True,
            "planned_precharged_and_actual_scored_multiplicity_separate": True,
            "finite_thresholds_required": True,
            "confirmation_slots_and_evidence_preallocated": True,
            "zero_survivors_allowed": True,
            "target_blind_definition_exact": True,
            "deterministic_no_clobber_receipt": True,
        },
        "idea_reservoir": {
            "reservoir_id": reservoir["reservoir_id"],
            "entries": len(reservoir["entries"]),
            "append_only": reservoir["append_only"],
            "genesis_sha256": reservoir["genesis_sha256"],
        },
        "campaign_governance": {
            "trusted_session_id": TRUSTED_SESSION_ID,
            "trusted_session_contract_sha256": trusted_session_contract_sha256(config),
            "terminal_ledger_schema_version": TERMINAL_LEDGER_SCHEMA,
            "terminal_ledger_path": TERMINAL_LEDGER_PATH.as_posix(),
            "terminal_ledger_contract_sha256": terminal_ledger_contract_sha256(config),
            "campaign_execution_authority_granted": False,
            "manifest_validation_scope": "STRUCTURAL_PREAUTHORIZATION_ONLY",
            "multiplicity_dimensions": list(MULTIPLICITY_DIMENSIONS),
            "candidate_domains": list(DOMAINS),
            "registered_concepts_are_not_scored_variants": True,
        },
        "access_accounting": {
            **zero_access,
            "governance_metadata_files_opened": len(ledger.opened),
            "governance_metadata_reads": ledger.rows(),
            "directory_enumerations": 0,
            "data_adapters_imported": 0,
        },
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() == payload:
                return "EXISTING_IDENTICAL"
            raise OpenGravityRegistryError(
                f"refusing to overwrite existing output: {path}"
            ) from None
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return "CREATED"
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_receipt(root: Path | None = None) -> str:
    repo = _repo_root() if root is None else root.resolve()
    payload = _canonical(build_receipt(repo)) + b"\n"
    return _atomic_no_clobber(repo / OUTPUT_PATH, payload)


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    stored = _load_json(repo / OUTPUT_PATH)
    validate_foundation_receipt(stored, repo)
    return stored


def validate_foundation_receipt(receipt: Mapping[str, Any], root: Path | None = None) -> None:
    repo = _repo_root() if root is None else root.resolve()
    expected = build_receipt(repo)
    if receipt != expected:
        raise OpenGravityRegistryError("stored foundation receipt does not match exact rebuild")
    payload = dict(receipt)
    observed = payload.pop("content_sha256", None)
    if observed != content_sha256(payload):
        raise OpenGravityRegistryError("stored foundation receipt content hash changed")


def status(root: Path | None = None) -> dict[str, Any]:
    receipt = check_receipt(root)
    return {
        "valid": True,
        "decision": receipt["decision"],
        "registry_id": receipt["registry_id"],
        "twell_concepts": receipt["twell_400_binding"]["total_concepts"],
        "ontology_nodes": receipt["registry"]["ontology_nodes"],
        "idea_reservoir_entries": receipt["idea_reservoir"]["entries"],
        "scientific_response_rows_opened": receipt["access_accounting"][
            "scientific_response_rows_opened"
        ],
        "scientific_scores_computed": receipt["access_accounting"]["scientific_scores_computed"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(json.dumps({"publication": write_receipt()}, sort_keys=True))
    elif args.command == "check":
        print(json.dumps(check_receipt(), sort_keys=True))
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
