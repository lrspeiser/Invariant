"""Sealed, response-blind primary-source comparator contract for open gravity.

This module reads only allowlisted governance metadata and implementation source/config
files.  It has no scientific-data adapter and cannot compute a scientific score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_primary_source_comparator_contract_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_primary_source_comparator_contract_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_primary_source_comparator_contract_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-primary-source-comparator-contract-v1/receipt.json")

CONTRACT_SCHEMA = "invariant-open-gravity-primary-source-comparator-contract-1.0"
RECEIPT_SCHEMA = "invariant-open-gravity-primary-source-comparator-contract-receipt-1.0"
CONTRACT_ID = "OPEN-GRAVITY-PRIMARY-SOURCE-COMPARATOR-CONTRACT-v1"
DECISION = (
    "PASS_METADATA_ONLY_PRIMARY_SOURCE_COMPARATOR_CONTRACT_ZERO_RESPONSE_ACCESS_NO_SCIENTIFIC_SCORE"
)
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "6b13fb6c32936aab13456455cc6160aae075da2c6bd02bec08ef4ff9bdf02a78"
)

EXPECTED_SECTION_SEALS = {
    "identity": "329e49e4a798f462c963354ddd3ce8eb9bc83bb06383b30dc54c33691a7cbb19",
    "purpose": "6db44d38f9d6484b898ba7139e6c25e812daf5289d63f1dcddd6c18073a52b13",
    "governance_bindings": ("cf6a058ac24762ef06ae0f68f30c2e48cf8b277872ba5c14fb3b29f70a78db30"),
    "access_contract": ("667a4bb6f2d03f532cce6fa97a873c328f077408b057d007b484053e5e882ca8"),
    "primary_sources": ("7d3286b3dc8c60476388829c83719e78fd8b310e80c999634ba84792524d6b3a"),
    "gp01_novelty_boundary": ("3d1da2b33a781fe12404dc4f82fa5c79b16d427325539d55c50149d515ec1685"),
    "dynamical_comparators": ("d848f662efa59d4de2895f712268ac60ff90f09886385429a9d8a62fcc95b131"),
    "ontology_prior_art": ("ebf15818b61e29af7d1e45a1648f33262b6b715763f9cc7d05ac53a7c982f481"),
    "light_gravity_analogies": ("c79cdf263f291fa2d01fcc1e7e1b19f15f39c4ce8235a3b5ec4c3bf025e62bde"),
    "implementation_bindings": ("c9c3f23d0af460d67715c2988e8fa88265e603154438181a892a7deca4b4f4bd"),
    "controlled_vocabularies": ("edfe2e9b8333735557a890c9cb1e1d84f79ecc61884036bdac0aeb3690ef471e"),
    "claim_boundary": ("84eefba42dc17d654f8cefd4bd3a44b6fa077d5721a9eb0fed7f840de5097366"),
    "output_path": ("c935f9340171ec6125d8607eed87e08f5a0138fbf3809745cf735724b6b5d9f6"),
}
EXPECTED_UNSEALED_ROOT_SHA256 = "56892bb85ef0afb0ec3a9ff77f1deee01e48c82a5a824efa6d8ae4b7087bb73c"

TOP_LEVEL_KEYS = frozenset(
    {
        *EXPECTED_SECTION_SEALS,
        "section_seals",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_IDS_REQUIRED = frozenset(
    {
        "SRC-GR-1916",
        "SRC-AQUAL-1984",
        "SRC-QUMOND-2010",
        "SRC-PENNER-2026",
        "SRC-RG-2016",
        "SRC-RG-COV-2023",
        "SRC-EMOND-2012",
        "SRC-MOG-2016",
        "SRC-NONLOCAL-2012",
        "SRC-NFW-1997",
        "SRC-EINASTO-2004",
        "SRC-GW150914-2016",
        "SRC-GW170817-2017",
        "SRC-WEINBERG-SPIN2-1964",
        "SRC-DONOGHUE-EFT-1994",
        "SRC-SQUEEZED-GRAVITON-1990",
        "SRC-FIERZ-PAULI-1939",
        "SRC-DRGT-2011",
        "SRC-EINSTEIN-AETHER-2001",
        "SRC-BOSE-ENTANGLEMENT-2017",
        "SRC-MARLETTO-VEDRAL-2017",
        "SRC-PAGE-GEILKER-1981",
        "SRC-HU-MATACZ-1995",
        "SRC-JACOBSON-THERMO-1995",
        "SRC-SAKHAROV-1967",
        "SRC-MALDACENA-1997",
        "SRC-VERLINDE-2011",
        "SRC-CAUSAL-SET-1987",
        "SRC-LOOP-GEOMETRY-1995",
        "SRC-CDT-2004",
        "SRC-UNRUH-1981",
    }
)
COMPARATOR_IDS_REQUIRED = frozenset(
    {
        "CMP-GR-BARYON",
        "CMP-EMPIRICAL-RAR",
        "CMP-AQUAL-ISOLATED",
        "CMP-AQUAL-EFE",
        "CMP-PENNER-2026",
        "CMP-REFRACTED-GRAVITY",
        "CMP-EMOND",
        "CMP-MOG-STVG",
        "CMP-MASHHOON-NONLOCAL",
        "CMP-GR-NFW",
        "CMP-GR-EINASTO",
    }
)
GP01_FINDING_IDS_REQUIRED = frozenset(
    {
        "GP01-LOCAL-AQUAL-BOUNDARY",
        "GP01-AQUAL-EFE-BOUNDARY",
        "GP01-T1-INTEGRABILITY-BOUNDARY",
        "GP01-T2-RELAXATION-BOUNDARY",
        "GP01-PDE-PERMITTIVITY-BOUNDARY",
        "GP01-HELMHOLTZ-NOT-TEMPORAL",
        "GP01-PENNER-NEIGHBOR",
        "GP01-ACTION-QUARANTINE",
    }
)
QG_IDS_REQUIRED = tuple(f"QG{index:02d}" for index in range(1, 14))
ANALOGY_IDS_REQUIRED = tuple(f"LG{index:02d}" for index in range(1, 14))
EXECUTION_STATUSES = frozenset(
    {
        "IMPLEMENTED_PREDECESSOR_REQUIRES_CAMPAIGN_REBIND",
        "SOLVER_BLOCKED",
        "SOURCE_AND_SOLVER_BLOCKED",
        "THEORY_ONLY_NOT_AN_EXECUTABLE_DYNAMICAL_COMPARATOR",
    }
)
SCIENTIFIC_STATUSES = frozenset(
    {
        "E_ESTABLISHED",
        "C_CONTROLLED_UNOBSERVED_SIGNATURE",
        "H_HYPOTHESIS",
        "A_ANALOGY",
    }
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
    "model_calls",
    "paid_calls",
)


class OpenGravityPriorArtError(RuntimeError):
    """Raised when a frozen primary-source/comparator invariant fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise OpenGravityPriorArtError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenGravityPriorArtError(f"nonempty string required: {label}")
    return value


def _require_unique_ids(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> set[str]:
    identifiers = [_require_nonempty_string(row.get(key), f"{label}.{key}") for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise OpenGravityPriorArtError(f"duplicate {label} identifiers")
    return set(identifiers)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OpenGravityPriorArtError(f"mapping required: {label}")
    return value


def _as_mapping_rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise OpenGravityPriorArtError(f"nonempty list required: {label}")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(value):
        rows.append(_as_mapping(row, f"{label}[{index}]"))
    return rows


def _validate_section_seals(config: Mapping[str, Any]) -> None:
    seals = _as_mapping(config.get("section_seals"), "section_seals")
    expected_keys = {*EXPECTED_SECTION_SEALS, "unsealed_root_sha256"}
    _require_exact_keys(seals, expected_keys, "section_seals")
    for section, expected in EXPECTED_SECTION_SEALS.items():
        recorded = seals.get(section)
        actual = content_sha256(config[section])
        if recorded != expected or actual != expected:
            raise OpenGravityPriorArtError(
                f"sealed section changed: {section}; recorded={recorded}, actual={actual}"
            )
    unsealed = {key: config[key] for key in config if key != "section_seals"}
    actual_root = content_sha256(unsealed)
    if (
        seals.get("unsealed_root_sha256") != EXPECTED_UNSEALED_ROOT_SHA256
        or actual_root != EXPECTED_UNSEALED_ROOT_SHA256
    ):
        raise OpenGravityPriorArtError("unsealed root changed")
    if content_sha256(config) != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise OpenGravityPriorArtError("canonical config hash changed")


def _validate_sources(config: Mapping[str, Any]) -> set[str]:
    rows = _as_mapping_rows(config["primary_sources"], "primary_sources")
    source_keys = {
        "source_id",
        "title",
        "authors",
        "exact_version",
        "url",
        "doi",
        "operational_anchor",
        "parameters_and_boundaries",
        "claimed_domain",
    }
    for index, row in enumerate(rows):
        _require_exact_keys(row, source_keys, f"primary_sources[{index}]")
        for key in source_keys:
            value = _require_nonempty_string(row[key], f"primary_sources[{index}].{key}")
            if key == "url" and not value.startswith("https://"):
                raise OpenGravityPriorArtError("primary-source URL must be exact HTTPS URL")
    identifiers = _require_unique_ids(rows, "source_id", "primary_sources")
    if identifiers != SOURCE_IDS_REQUIRED:
        raise OpenGravityPriorArtError("primary-source inventory changed")
    return identifiers


def _validate_gp01(config: Mapping[str, Any], source_ids: set[str]) -> None:
    boundary = _as_mapping(config["gp01_novelty_boundary"], "gp01_novelty_boundary")
    _require_exact_keys(
        boundary,
        {
            "historical_novelty_status",
            "frozen_findings",
            "no_negative_search_inference",
            "human_review_required_before_novelty_language",
        },
        "gp01_novelty_boundary",
    )
    if (
        boundary["historical_novelty_status"] != "OPEN_REQUIRES_INDEPENDENT_HUMAN_PRIOR_ART_REVIEW"
        or boundary["human_review_required_before_novelty_language"] is not True
    ):
        raise OpenGravityPriorArtError("GP01 novelty gate was weakened")
    rows = _as_mapping_rows(boundary["frozen_findings"], "frozen_findings")
    finding_keys = {
        "finding_id",
        "gp01_branch",
        "relationship",
        "exact_statement",
        "source_ids",
        "promotion_consequence",
    }
    for index, row in enumerate(rows):
        _require_exact_keys(row, finding_keys, f"frozen_findings[{index}]")
        references = row["source_ids"]
        if not isinstance(references, list) or not references or not set(references) <= source_ids:
            raise OpenGravityPriorArtError("GP01 finding has missing or unknown source anchor")
    if _require_unique_ids(rows, "finding_id", "frozen_findings") != GP01_FINDING_IDS_REQUIRED:
        raise OpenGravityPriorArtError("GP01 novelty-boundary finding inventory changed")


def _validate_comparators(config: Mapping[str, Any], source_ids: set[str]) -> None:
    rows = _as_mapping_rows(config["dynamical_comparators"], "dynamical_comparators")
    comparator_keys = {
        "comparator_id",
        "display_name",
        "source_ids",
        "exact_equation_or_definition",
        "parameters_and_boundaries",
        "claimed_domain",
        "closest_equivalence",
        "sparc_status",
        "sparc_missing_requirements",
        "xcop_status",
        "xcop_missing_requirements",
        "never_substitute_name_only",
    }
    for index, row in enumerate(rows):
        _require_exact_keys(row, comparator_keys, f"dynamical_comparators[{index}]")
        references = row["source_ids"]
        if not isinstance(references, list) or not references or not set(references) <= source_ids:
            raise OpenGravityPriorArtError("comparator has missing or unknown source anchor")
        if row["never_substitute_name_only"] is not True:
            raise OpenGravityPriorArtError("name-only comparator substitution was enabled")
        for domain in ("sparc", "xcop"):
            status = row[f"{domain}_status"]
            missing = row[f"{domain}_missing_requirements"]
            if status not in EXECUTION_STATUSES or not isinstance(missing, list):
                raise OpenGravityPriorArtError(f"invalid {domain} comparator status")
            blocked = status in {"SOLVER_BLOCKED", "SOURCE_AND_SOLVER_BLOCKED"}
            if blocked != bool(missing):
                raise OpenGravityPriorArtError(
                    f"{domain} blocked status and missing requirements disagree"
                )
            if status == "THEORY_ONLY_NOT_AN_EXECUTABLE_DYNAMICAL_COMPARATOR":
                raise OpenGravityPriorArtError("theory-only status is invalid in comparator table")
    if _require_unique_ids(rows, "comparator_id", "comparators") != COMPARATOR_IDS_REQUIRED:
        raise OpenGravityPriorArtError("dynamical comparator inventory changed")


def _validate_ontologies_and_analogies(config: Mapping[str, Any], source_ids: set[str]) -> None:
    ontology_rows = _as_mapping_rows(config["ontology_prior_art"], "ontology_prior_art")
    ontology_keys = {
        "ontology_id",
        "name",
        "scientific_status",
        "source_ids",
        "operational_definition",
        "parameters_and_boundaries",
        "claimed_domain",
        "closest_equivalence",
        "sparc_status",
        "xcop_status",
        "radial_substitution_rule",
    }
    for index, row in enumerate(ontology_rows):
        _require_exact_keys(row, ontology_keys, f"ontology_prior_art[{index}]")
        if row["scientific_status"] not in SCIENTIFIC_STATUSES:
            raise OpenGravityPriorArtError("invalid ontology scientific status")
        references = row["source_ids"]
        if not isinstance(references, list) or not set(references) <= source_ids:
            raise OpenGravityPriorArtError("ontology has unknown source anchor")
        for domain in ("sparc", "xcop"):
            if row[f"{domain}_status"] not in EXECUTION_STATUSES:
                raise OpenGravityPriorArtError("invalid ontology execution status")
        if row["ontology_id"] != "QG01" and (
            row["sparc_status"] != "THEORY_ONLY_NOT_AN_EXECUTABLE_DYNAMICAL_COMPARATOR"
            or row["xcop_status"] != "THEORY_ONLY_NOT_AN_EXECUTABLE_DYNAMICAL_COMPARATOR"
        ):
            raise OpenGravityPriorArtError("ontology family was promoted to a radial comparator")
    if tuple(row["ontology_id"] for row in ontology_rows) != QG_IDS_REQUIRED:
        raise OpenGravityPriorArtError("QG01-QG13 inventory or order changed")

    analogy_rows = _as_mapping_rows(config["light_gravity_analogies"], "light_gravity_analogies")
    analogy_keys = {
        "analogy_id",
        "light_concept",
        "gravity_analogue",
        "source_ids",
        "operational_seed",
        "required_caution",
        "executable_status",
    }
    expected_status = "ANALOGY_ONLY_SOURCE_BLOCKED_UNTIL_TYPED_EQUATIONS_SOLVER_AND_BOUNDARIES"
    for index, row in enumerate(analogy_rows):
        _require_exact_keys(row, analogy_keys, f"light_gravity_analogies[{index}]")
        references = row["source_ids"]
        if not isinstance(references, list) or not references or not set(references) <= source_ids:
            raise OpenGravityPriorArtError("analogy has missing or unknown source anchor")
        if row["executable_status"] != expected_status:
            raise OpenGravityPriorArtError("analogy was promoted without a typed solver")
    if tuple(row["analogy_id"] for row in analogy_rows) != ANALOGY_IDS_REQUIRED:
        raise OpenGravityPriorArtError("LG01-LG13 analogy inventory or order changed")


def _validate_implementation_bindings(config: Mapping[str, Any]) -> None:
    rows = _as_mapping_rows(config["implementation_bindings"], "implementation_bindings")
    binding_keys = {
        "binding_id",
        "config_path",
        "config_sha256",
        "module_path",
        "module_sha256",
        "implemented_comparator_ids",
        "campaign_rebind_required",
        "response_receipt_read_forbidden",
    }
    for index, row in enumerate(rows):
        _require_exact_keys(row, binding_keys, f"implementation_bindings[{index}]")
        for key in ("config_sha256", "module_sha256"):
            if SHA256_RE.fullmatch(str(row[key])) is None:
                raise OpenGravityPriorArtError("invalid implementation SHA-256")
        if (
            row["campaign_rebind_required"] is not True
            or row["response_receipt_read_forbidden"] is not True
        ):
            raise OpenGravityPriorArtError("implementation predecessor gate was weakened")
        implemented = row["implemented_comparator_ids"]
        if not isinstance(implemented, list) or not implemented:
            raise OpenGravityPriorArtError("empty implementation comparator binding")
        if not set(implemented) <= COMPARATOR_IDS_REQUIRED:
            raise OpenGravityPriorArtError("unknown implementation comparator binding")
    _require_unique_ids(rows, "binding_id", "implementation_bindings")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate every frozen section and fail closed on any nested mutation."""

    _require_exact_keys(config, set(TOP_LEVEL_KEYS), "config")
    _validate_section_seals(config)
    identity = _as_mapping(config["identity"], "identity")
    _require_exact_keys(
        identity,
        {
            "schema_version",
            "contract_id",
            "semantic_version",
            "status",
            "append_only",
            "frozen_cutoff_utc",
        },
        "identity",
    )
    if (
        identity["schema_version"] != CONTRACT_SCHEMA
        or identity["contract_id"] != CONTRACT_ID
        or identity["semantic_version"] != "1.0.0"
        or identity["append_only"] is not True
    ):
        raise OpenGravityPriorArtError("contract identity changed")
    access = _as_mapping(config["access_contract"], "access_contract")
    zero = _as_mapping(access.get("zero_access"), "access_contract.zero_access")
    _require_exact_keys(zero, set(ZERO_ACCESS_FIELDS), "zero_access")
    if any(zero[field_name] != 0 for field_name in ZERO_ACCESS_FIELDS):
        raise OpenGravityPriorArtError("response/model/score zero-access contract changed")
    for field_name in (
        "receipt_rebuild_network_calls",
        "receipt_rebuild_model_calls",
        "receipt_rebuild_paid_calls",
    ):
        if access.get(field_name) != 0:
            raise OpenGravityPriorArtError("receipt rebuild external-call count is nonzero")
    source_ids = _validate_sources(config)
    _validate_gp01(config, source_ids)
    _validate_comparators(config, source_ids)
    _validate_ontologies_and_analogies(config, source_ids)
    _validate_implementation_bindings(config)
    vocab = _as_mapping(config["controlled_vocabularies"], "controlled_vocabularies")
    if set(vocab.get("execution_statuses", [])) != EXECUTION_STATUSES:
        raise OpenGravityPriorArtError("execution vocabulary changed")
    if set(vocab.get("scientific_statuses", [])) != SCIENTIFIC_STATUSES:
        raise OpenGravityPriorArtError("scientific-status vocabulary changed")
    if "family name" not in str(vocab.get("source_block_rule", "")):
        raise OpenGravityPriorArtError("family-name fail-closed rule missing")
    if config["output_path"] != OUTPUT_PATH.as_posix():
        raise OpenGravityPriorArtError("receipt output path changed")


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    try:
        value = json.loads((repo / CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenGravityPriorArtError("could not load frozen contract config") from error
    if not isinstance(value, dict):
        raise OpenGravityPriorArtError("contract config must be a JSON object")
    validate_config(value)
    return value


@dataclass
class MetadataAccessLedger:
    """Allowlist and record every metadata read used to rebuild the receipt."""

    repo: Path
    allowed: Mapping[Path, str]
    opened: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._normalized = {path.resolve(): kind for path, kind in self.allowed.items()}

    def display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo).as_posix()
        except ValueError:
            return path.as_posix()

    def read_bytes(self, path: Path) -> bytes:
        resolved = path.resolve()
        if resolved not in self._normalized:
            raise OpenGravityPriorArtError(f"non-allowlisted read refused: {resolved}")
        display = self.display_path(resolved)
        self.opened[display] = self._normalized[resolved]
        try:
            return resolved.read_bytes()
        except OSError as error:
            raise OpenGravityPriorArtError(f"could not read metadata: {resolved}") from error

    def rows(self) -> list[dict[str, str]]:
        return [{"path": path, "artifact_kind": self.opened[path]} for path in sorted(self.opened)]


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenGravityPriorArtError(f"invalid JSON metadata: {label}") from error
    if not isinstance(value, dict):
        raise OpenGravityPriorArtError(f"JSON object required: {label}")
    return value


def _absolute_or_repo(repo: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo / path


def _allowed_metadata(repo: Path, config: Mapping[str, Any]) -> dict[Path, str]:
    allowed = {
        repo / CONFIG_PATH: "sealed_contract_config",
        repo / MODULE_PATH: "sealed_contract_module",
        repo / TEST_PATH: "sealed_contract_tests",
    }
    governance = _as_mapping(config["governance_bindings"], "governance_bindings")
    for binding_id, binding in governance.items():
        row = _as_mapping(binding, f"governance_bindings.{binding_id}")
        allowed[_absolute_or_repo(repo, str(row["path"]))] = f"governance:{binding_id}"
    for binding in config["implementation_bindings"]:
        allowed[repo / str(binding["config_path"])] = (
            f"implementation_config:{binding['binding_id']}"
        )
        allowed[repo / str(binding["module_path"])] = (
            f"implementation_module:{binding['binding_id']}"
        )
    return allowed


def _verify_bound_hashes(
    repo: Path, config: Mapping[str, Any], ledger: MetadataAccessLedger
) -> dict[str, str]:
    verified: dict[str, str] = {}
    governance = _as_mapping(config["governance_bindings"], "governance_bindings")
    for binding_id, binding in governance.items():
        row = _as_mapping(binding, f"governance_bindings.{binding_id}")
        path = _absolute_or_repo(repo, str(row["path"]))
        actual = _sha256_bytes(ledger.read_bytes(path))
        if actual != row["sha256"]:
            raise OpenGravityPriorArtError(f"governance binding changed: {binding_id}")
        verified[ledger.display_path(path.resolve())] = actual
    for binding in config["implementation_bindings"]:
        for kind in ("config", "module"):
            path = repo / str(binding[f"{kind}_path"])
            actual = _sha256_bytes(ledger.read_bytes(path))
            if actual != binding[f"{kind}_sha256"]:
                raise OpenGravityPriorArtError(
                    f"implementation binding changed: {binding['binding_id']}:{kind}"
                )
            verified[ledger.display_path(path.resolve())] = actual
    return dict(sorted(verified.items()))


def receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_content_sha256", None)
    return content_sha256(payload)


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    """Rebuild the deterministic receipt from allowlisted metadata only."""

    repo = _repo_root() if root is None else root.resolve()
    config_path = repo / CONFIG_PATH
    try:
        raw_config = config_path.read_bytes()
    except OSError as error:
        raise OpenGravityPriorArtError("could not read contract config") from error
    config = _json_object(raw_config, str(config_path))
    validate_config(config)

    ledger = MetadataAccessLedger(repo, _allowed_metadata(repo, config))
    # Re-read through the ledger so the audit records the only config access used by the receipt.
    recorded_config = ledger.read_bytes(config_path)
    if recorded_config != raw_config:
        raise OpenGravityPriorArtError("contract config changed during receipt rebuild")
    bound_hashes = _verify_bound_hashes(repo, config, ledger)
    module_hash = _sha256_bytes(ledger.read_bytes(repo / MODULE_PATH))
    test_hash = _sha256_bytes(ledger.read_bytes(repo / TEST_PATH))

    comparators = config["dynamical_comparators"]
    comparator_matrix = [
        {
            "comparator_id": row["comparator_id"],
            "sparc_status": row["sparc_status"],
            "xcop_status": row["xcop_status"],
        }
        for row in comparators
    ]
    status_counts = {
        domain: dict(sorted(Counter(row[f"{domain}_status"] for row in comparators).items()))
        for domain in ("sparc", "xcop")
    }
    source_blocked = sorted(
        row["comparator_id"]
        for row in comparators
        if "BLOCKED" in row["sparc_status"] or "BLOCKED" in row["xcop_status"]
    )
    zero = dict(config["access_contract"]["zero_access"])
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "semantic_version": "1.0.0",
        "decision": DECISION,
        "config_canonical_sha256": content_sha256(config),
        "config_file_sha256": _sha256_bytes(raw_config),
        "section_seals": dict(config["section_seals"]),
        "artifact_hashes": {
            "module_sha256": module_hash,
            "test_sha256": test_hash,
            "bound_metadata_sha256": bound_hashes,
        },
        "source_inventory": {
            "primary_source_count": len(config["primary_sources"]),
            "source_ids": [row["source_id"] for row in config["primary_sources"]],
            "exact_version_and_url_frozen_for_every_source": True,
        },
        "gp01_novelty_boundary": {
            "historical_novelty_status": config["gp01_novelty_boundary"][
                "historical_novelty_status"
            ],
            "finding_ids": [
                row["finding_id"] for row in config["gp01_novelty_boundary"]["frozen_findings"]
            ],
            "human_review_required_before_novelty_language": True,
        },
        "comparator_inventory": {
            "comparator_count": len(comparators),
            "execution_matrix": comparator_matrix,
            "status_counts": status_counts,
            "blocked_comparator_ids": source_blocked,
            "family_name_is_never_an_executable_substitute": True,
            "implemented_predecessors_require_campaign_rebind": True,
        },
        "ontology_and_analogy_inventory": {
            "ontology_ids": [row["ontology_id"] for row in config["ontology_prior_art"]],
            "light_gravity_analogy_ids": [
                row["analogy_id"] for row in config["light_gravity_analogies"]
            ],
            "non_qg01_ontologies_are_not_radial_comparators": True,
            "analogies_are_source_blocked_until_typed": True,
        },
        "access_audit": {
            **zero,
            "network_calls_during_receipt_rebuild": 0,
            "model_calls_during_receipt_rebuild": 0,
            "paid_calls_during_receipt_rebuild": 0,
            "response_bearing_receipts_opened": 0,
            "allowlisted_metadata_files_opened": ledger.rows(),
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    receipt["receipt_content_sha256"] = receipt_content_sha256(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    """Atomically create ``path`` and fail if it already exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise OpenGravityPriorArtError(
                f"refusing to overwrite existing receipt: {path}"
            ) from error
        except OSError as error:
            if path.exists():
                raise OpenGravityPriorArtError(
                    f"refusing to overwrite existing receipt: {path}"
                ) from error
            raise OpenGravityPriorArtError(f"atomic receipt creation failed: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def write_receipt(receipt: Mapping[str, Any], output: Path) -> None:
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_no_clobber(output, payload)


def check_receipt(root: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    target = repo / OUTPUT_PATH if output is None else output.resolve()
    expected = build_receipt(repo)
    try:
        stored = _json_object(target.read_bytes(), str(target))
    except OSError as error:
        raise OpenGravityPriorArtError(f"could not read stored receipt: {target}") from error
    if stored != expected:
        raise OpenGravityPriorArtError("stored receipt does not match deterministic rebuild")
    if stored.get("receipt_content_sha256") != receipt_content_sha256(stored):
        raise OpenGravityPriorArtError("stored receipt content hash is invalid")
    return stored


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="atomically create the frozen receipt")
    build.add_argument("--output", type=Path, default=OUTPUT_PATH)
    check = subparsers.add_parser("check", help="verify the stored receipt")
    check.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = _repo_root()
    try:
        if args.command == "build":
            receipt = build_receipt(repo)
            output = args.output if args.output.is_absolute() else repo / args.output
            write_receipt(receipt, output)
        else:
            output = args.output if args.output.is_absolute() else repo / args.output
            receipt = check_receipt(repo, output)
    except OpenGravityPriorArtError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
