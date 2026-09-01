"""Target-blind typed compiler for the frozen TWELL-400-v2 ontology.

Only governance, prior-art, source-availability, the retained failed card stream, and
this packet's own files are read.  The pending central schema is deliberately not read
or hash-bound.  No astronomy response adapter or scientific score exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/twell_400_v2_typed_compiler_packet_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/twell_400_v2_typed_compiler_packet.py")
TEST_PATH = Path("tests/test_twell_400_v2_typed_compiler_packet.py")
CARDS_PATH = Path("runs/gravity/twell-400-v2-typed-compiler-packet-v1/cards-repair-v1.jsonl")
RECEIPT_PATH = Path("runs/gravity/twell-400-v2-typed-compiler-packet-v1/receipt-repair-v1.json")
FAILED_CARDS_PATH = Path("runs/gravity/twell-400-v2-typed-compiler-packet-v1/cards.jsonl")

CONFIG_SCHEMA = "invariant-twell-400-v2-typed-compiler-packet-1.0"
CARD_SCHEMA = "invariant-open-gravity-mechanism-card-1.0"
RECEIPT_SCHEMA = "invariant-twell-400-v2-typed-compiler-packet-receipt-1.1"
PACKET_ID = "TWELL-400-v2-TYPED-COMPILER-PACKET-v1"
DECISION = (
    "PASS_INTERIM_TWELL_400_V2_EXACT_CELL_COMPILER_DEFERRED_BINDINGS_"
    "ZERO_RESPONSE_ACCESS_NO_CAMPAIGN_AUTHORITY"
)
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "0bc6dbe7c0514fbe6c8058d3315ff3efb56893605fe9868ce9674ef9edd36909"
)
EXPECTED_UNSEALED_ROOT_SHA256 = "c3bc95a6059eecd20c7a1ecde2bbd0014b211c5779fed8580498f472b2a034ae"
EXPECTED_SECTION_SEALS = {
    "identity": "2226ed957cf07d488d52c9a173a9e67ef8b36b4c5d4a2c19d2c6ee02c5f5c523",
    "governance_bindings": ("b7adec8590d170daa30bec46e2d17f327db7a1357da267061ed34c027396ecff"),
    "access_contract": ("607afdcf2e62ced547ac671291f2e475eddac3adc39c8fb55226d97612646c54"),
    "driver_catalog": ("484fec5c2097c72ae30ce68d85c87f1145ab0d37e03e311af29435de31ffde3e"),
    "architecture_catalog": ("9a7b1b7620fbeb69c3b2a45f4113b879dc7a0d9dcd0670d3c2b3a94bab0c5d1a"),
    "compound_catalog": ("90c6d9ed9fc52fe710fe440ed2983ee4895d6866df1140180675ba5d188c89ab"),
    "compiler_contract": ("5a1c2e00aae1475306da715d2c13a23c4b6a98b4313d4898cd2d0dd38b59c372"),
    "probe_contract": ("10f39260b2fc1e1939948463c9eac3bfed80f084da4568ad7a2aa9de7b1bb88c"),
    "status_contract": ("f706c53f57ce7d1903022d10db3d48faf729158067d9f54009f20cfdd86d6740"),
    "output_contract": ("1a5f9c1e24a3c57d0a132548e4338a14ece7374638584401640a8ff1edd3c556"),
    "claim_boundary": ("6d5db8a1e0b17ab76c558856d23b08abd2b7b14934ccda5da06c15cc42d8bf03"),
}
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
SOURCE_READY_PREFIX = "SOURCE_AVAILABLE"
FORBIDDEN_RESPONSE_INPUTS = [
    "motion_response",
    "pressure_response",
    "temperature_response",
    "lensing_response",
    "frequency_or_redshift_response",
    "inferred_total_mass",
    "residuals",
    "partial_rankings",
    "confirmation_rows",
    "independent_rows",
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TwellCompilerError(RuntimeError):
    """Raised when a frozen compiler invariant fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def ordered_concept_ids() -> list[str]:
    atomic = [
        f"TW2-A{architecture:02d}-D{driver:02d}"
        for architecture in range(1, 20)
        for driver in range(1, 21)
    ]
    return atomic + [f"X{compound:02d}" for compound in range(1, 21)]


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise TwellCompilerError(
            f"{label} keys changed; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TwellCompilerError(f"mapping required: {label}")
    return value


def _rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TwellCompilerError(f"nonempty list required: {label}")
    return [_mapping(row, f"{label}[{index}]") for index, row in enumerate(value)]


def _unique_ids(rows: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    identifiers = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise TwellCompilerError(f"invalid or duplicate IDs: {label}")
    return identifiers


def validate_config(config: Mapping[str, Any]) -> None:
    top_keys = {*EXPECTED_SECTION_SEALS, "section_seals"}
    _require_exact_keys(config, top_keys, "config")
    seals = _mapping(config["section_seals"], "section_seals")
    _require_exact_keys(seals, {*EXPECTED_SECTION_SEALS, "unsealed_root_sha256"}, "seals")
    for section, expected in EXPECTED_SECTION_SEALS.items():
        if seals[section] != expected or content_sha256(config[section]) != expected:
            raise TwellCompilerError(f"sealed section changed: {section}")
    unsealed = {key: value for key, value in config.items() if key != "section_seals"}
    if (
        seals["unsealed_root_sha256"] != EXPECTED_UNSEALED_ROOT_SHA256
        or content_sha256(unsealed) != EXPECTED_UNSEALED_ROOT_SHA256
    ):
        raise TwellCompilerError("unsealed config root changed")
    if content_sha256(config) != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise TwellCompilerError("canonical config hash changed")

    identity = _mapping(config["identity"], "identity")
    if (
        identity.get("schema_version") != CONFIG_SCHEMA
        or identity.get("packet_id") != PACKET_ID
        or identity.get("append_only") is not True
        or identity.get("frozen_before_response_access") is not True
    ):
        raise TwellCompilerError("packet identity changed")
    drivers = _rows(config["driver_catalog"], "driver_catalog")
    architectures = _rows(config["architecture_catalog"], "architecture_catalog")
    compounds = _rows(config["compound_catalog"], "compound_catalog")
    expected_drivers = [
        f"D{index:02d}_{suffix}"
        for index, suffix in enumerate(
            [
                "ACC",
                "POT",
                "RAD",
                "RHO",
                "SIG",
                "SLOPE",
                "TIDE",
                "BAL",
                "FLAT",
                "MULT",
                "EDGE",
                "ENV",
                "GASF",
                "ION",
                "COOL",
                "NTH",
                "AGE",
                "RELAX",
                "COH",
                "EPOCH",
            ],
            start=1,
        )
    ]
    expected_architectures = [
        f"A{index:02d}_{suffix}"
        for index, suffix in enumerate(
            [
                "LAPSE",
                "CLOCK",
                "CONFORMAL",
                "DISFORMAL",
                "SLIP",
                "SPATIAL_KERNEL",
                "BOUNDARY",
                "PERMITTIVITY",
                "ENTROPIC",
                "DENSITY_SCREEN",
                "DERIV_SCREEN",
                "MASSIVE",
                "MIXED_MODE",
                "PHASE",
                "RETARDED",
                "MEMORY",
                "RESONANCE",
                "STOCHASTIC",
                "FEEDBACK",
            ],
            start=1,
        )
    ]
    if _unique_ids(drivers, "drivers") != expected_drivers:
        raise TwellCompilerError("20-driver inventory or order changed")
    if _unique_ids(architectures, "architectures") != expected_architectures:
        raise TwellCompilerError("19-architecture inventory or order changed")
    if _unique_ids(compounds, "compounds") != [f"X{i:02d}" for i in range(1, 21)]:
        raise TwellCompilerError("20-compound inventory or order changed")
    for row in drivers:
        if float(row["reference_value"]) <= 0 or "u=" not in str(row["normalized_expression"]):
            raise TwellCompilerError("driver dimension/normalization contract incomplete")
    driver_ids = {row["id"] for row in drivers}
    architecture_ids = {row["id"] for row in architectures}
    for row in architectures:
        if (
            not row.get("canonical_expressions")
            or not row.get("parameters")
            or not row.get("initial_conditions")
            or not row.get("boundaries")
            or not row.get("source_needs")
            or row.get("executable_template") is not True
        ):
            raise TwellCompilerError(f"incomplete architecture template: {row['id']}")
        parameter_names = [str(item.get("name", "")) for item in row["parameters"]]
        if (
            "lambda" not in parameter_names
            or len(parameter_names) != len(set(parameter_names))
            or any(
                not item.get("values")
                or not item.get("unit")
                or any(not math.isfinite(float(value)) for value in item["values"])
                for item in row["parameters"]
            )
        ):
            raise TwellCompilerError(f"invalid architecture parameter grid: {row['id']}")
    for row in compounds:
        if (
            len(row.get("drivers", [])) != 2
            or not set(row["drivers"]) <= driver_ids
            or row.get("architecture") not in architecture_ids
            or not row.get("operation")
            or not row.get("operation_id")
        ):
            raise TwellCompilerError(f"incomplete compound operation: {row['id']}")
    architecture_cells = {
        row["id"]: math.prod(len(parameter["values"]) for parameter in row["parameters"])
        for row in architectures
    }
    atomic_parameter_cells = 20 * sum(architecture_cells.values())
    compound_parameter_cells = sum(architecture_cells[row["architecture"]] for row in compounds)
    override_cells = sum(row["id"] in {"X19", "X20"} for row in compounds)
    probe = _mapping(config["probe_contract"], "probe_contract")
    if (
        atomic_parameter_cells + compound_parameter_cells
        != probe.get("base_cartesian_parameter_cell_count")
        or override_cells != probe.get("compound_override_evidence_cell_count")
        or atomic_parameter_cells + compound_parameter_cells + override_cells
        != probe.get("expected_concept_parameter_cell_count")
    ):
        raise TwellCompilerError("concept-parameter Cartesian grid contract changed")
    compiler = _mapping(config["compiler_contract"], "compiler_contract")
    if (
        compiler.get("total_count") != 400
        or compiler.get("atomic_count") != 380
        or compiler.get("compound_count") != 20
        or compiler.get("default_photon_closure") != "L0_NO_LIGHT_CLAIM"
        or compiler.get("default_capture_closure") != "C0_ISOLATED_CONSERVATIVE"
        or compiler.get("campaign_freeze_authority") is not False
        or content_sha256(ordered_concept_ids()) != compiler.get("ordered_concept_ids_sha256")
    ):
        raise TwellCompilerError("compiler enumeration or closure contract changed")
    access = _mapping(config["access_contract"], "access_contract")
    zero = _mapping(access.get("zero_access"), "zero_access")
    _require_exact_keys(zero, set(ZERO_ACCESS_FIELDS), "zero_access")
    if any(zero[field_name] != 0 for field_name in ZERO_ACCESS_FIELDS):
        raise TwellCompilerError("zero-access contract changed")
    governance = _mapping(config["governance_bindings"], "governance_bindings")
    for row in governance["deferred_informational"]:
        if "sha256" in row or row["may_authorize_campaign"] is not False:
            raise TwellCompilerError("deferred registry/GP01 binding was hardened or authorized")
    if governance["campaign_manifest_frozen"] is not False:
        raise TwellCompilerError("compiler cannot freeze a campaign")


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    try:
        value = json.loads((repo / CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TwellCompilerError("could not load typed compiler config") from error
    if not isinstance(value, dict):
        raise TwellCompilerError("compiler config must be a JSON object")
    validate_config(value)
    return value


@dataclass
class MetadataLedger:
    repo: Path
    allowed: Mapping[Path, str]
    opened: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.normalized = {path.resolve(): kind for path, kind in self.allowed.items()}

    def display(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo).as_posix()
        except ValueError:
            return path.as_posix()

    def read_bytes(self, path: Path) -> bytes:
        resolved = path.resolve()
        if resolved not in self.normalized:
            raise TwellCompilerError(f"non-allowlisted metadata read refused: {resolved}")
        self.opened[self.display(resolved)] = self.normalized[resolved]
        try:
            return resolved.read_bytes()
        except OSError as error:
            raise TwellCompilerError(f"could not read metadata: {resolved}") from error

    def rows(self) -> list[dict[str, str]]:
        return [{"path": path, "artifact_kind": self.opened[path]} for path in sorted(self.opened)]


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TwellCompilerError(f"invalid JSON metadata: {label}") from error
    if not isinstance(value, dict):
        raise TwellCompilerError(f"JSON object required: {label}")
    return value


def _absolute_or_repo(repo: Path, text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else repo / path


def _bound_inputs(
    repo: Path, config: Mapping[str, Any]
) -> tuple[MetadataLedger, dict[str, bytes], bytes, bytes, bytes, bytes]:
    hard = config["governance_bindings"]["hard_bound"]
    allowed: dict[Path, str] = {
        repo / CONFIG_PATH: "typed_compiler_config",
        repo / MODULE_PATH: "typed_compiler_module",
        repo / TEST_PATH: "typed_compiler_tests",
        repo / FAILED_CARDS_PATH: "failed_compiler_card_stream_counterevidence",
    }
    for row in hard:
        allowed[_absolute_or_repo(repo, row["path"])] = f"hard_binding:{row['binding_id']}"
    ledger = MetadataLedger(repo, allowed)
    raw_config = ledger.read_bytes(repo / CONFIG_PATH)
    if _json_object(raw_config, str(CONFIG_PATH)) != config:
        raise TwellCompilerError("compiler config changed during rebuild")
    payloads: dict[str, bytes] = {}
    for row in hard:
        payload = ledger.read_bytes(_absolute_or_repo(repo, row["path"]))
        if _sha256_bytes(payload) != row["sha256"]:
            raise TwellCompilerError(f"hard-bound input changed: {row['binding_id']}")
        payloads[row["binding_id"]] = payload
    module_bytes = ledger.read_bytes(repo / MODULE_PATH)
    test_bytes = ledger.read_bytes(repo / TEST_PATH)
    failed_cards_bytes = ledger.read_bytes(repo / FAILED_CARDS_PATH)
    failed_binding = config["output_contract"]["failed_packet_retained"]
    if _sha256_bytes(failed_cards_bytes) != failed_binding["cards_sha256"]:
        raise TwellCompilerError("failed compiler card stream counterevidence changed")
    return ledger, payloads, raw_config, module_bytes, test_bytes, failed_cards_bytes


def _local_ref(schema: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise TwellCompilerError(f"only local schema references supported: {reference}")
    value: Any = schema
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            raise TwellCompilerError(f"broken schema reference: {reference}")
        value = value[token]
    return _mapping(value, reference)


def schema_errors(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Validate the strict JSON-Schema subset used by mechanism cards."""

    errors: list[str] = []

    def type_matches(value: Any, expected: str) -> bool:
        return {
            "object": isinstance(value, Mapping),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "null": value is None,
        }[expected]

    def visit(value: Any, rule: Mapping[str, Any], path: str) -> None:
        if "$ref" in rule:
            visit(value, _local_ref(schema, str(rule["$ref"])), path)
            return
        if "oneOf" in rule:
            matches = 0
            for branch in rule["oneOf"]:
                before = len(errors)
                visit(value, branch, path)
                if len(errors) == before:
                    matches += 1
                else:
                    del errors[before:]
            if matches != 1:
                errors.append(f"{path}: oneOf match count {matches}")
            return
        if "const" in rule and _canonical(value) != _canonical(rule["const"]):
            errors.append(f"{path}: constant mismatch")
            return
        if "enum" in rule and not any(
            _canonical(value) == _canonical(item) for item in rule["enum"]
        ):
            errors.append(f"{path}: enum mismatch")
            return
        expected_type = rule.get("type")
        if expected_type is not None:
            choices = [expected_type] if isinstance(expected_type, str) else list(expected_type)
            if not any(type_matches(value, choice) for choice in choices):
                errors.append(f"{path}: type mismatch")
                return
        if isinstance(value, Mapping):
            required = set(rule.get("required", []))
            if required - set(value):
                errors.append(f"{path}: missing {sorted(required - set(value))}")
            properties = rule.get("properties", {})
            if rule.get("additionalProperties") is False and set(value) - set(properties):
                errors.append(f"{path}: extra {sorted(set(value) - set(properties))}")
            for key, child in properties.items():
                if key in value:
                    visit(value[key], child, f"{path}.{key}")
        if isinstance(value, list):
            if len(value) < int(rule.get("minItems", 0)):
                errors.append(f"{path}: too few items")
            if rule.get("uniqueItems") and len({_canonical(item) for item in value}) != len(value):
                errors.append(f"{path}: duplicate items")
            if isinstance(rule.get("items"), Mapping):
                for index, item in enumerate(value):
                    visit(item, rule["items"], f"{path}[{index}]")
        if isinstance(value, str):
            if len(value) < int(rule.get("minLength", 0)):
                errors.append(f"{path}: too short")
            if "pattern" in rule and re.fullmatch(str(rule["pattern"]), value) is None:
                errors.append(f"{path}: pattern mismatch")
            if rule.get("format") == "date-time":
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    errors.append(f"{path}: invalid date-time")
                else:
                    if parsed.tzinfo is None:
                        errors.append(f"{path}: date-time lacks offset")

    visit(instance, schema, "$")
    return errors


def _validate_bound_metadata(
    config: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any]]:
    prior_art = _json_object(payloads["PRIMARY-SOURCE-PRIOR-ART"], "prior art")
    source = _json_object(payloads["SOURCE-AVAILABILITY"], "source availability")
    prior_ids = {row["source_id"] for row in prior_art["primary_sources"]}
    for architecture in config["architecture_catalog"]:
        if not set(architecture["prior_art_source_ids"]) <= prior_ids:
            raise TwellCompilerError(f"unknown prior-art source: {architecture['id']}")
    source_drivers = source["driver_source_availability"]
    for driver in config["driver_catalog"]:
        for domain, key in (("SPARC", "sparc_source_status"), ("XCOP", "xcop_source_status")):
            if driver[key] != source_drivers[domain][driver["id"]]:
                raise TwellCompilerError(f"source snapshot changed: {domain}:{driver['id']}")
    source_compounds = source["mechanism_registry"]["twell"]["compounds"]
    frozen_tuples = [
        (row["id"], row["drivers"], row["architecture"]) for row in config["compound_catalog"]
    ]
    source_tuples = [(row["id"], row["drivers"], row["architecture"]) for row in source_compounds]
    if frozen_tuples != source_tuples:
        raise TwellCompilerError("compound source/architecture binding changed")
    return prior_art, source


def _fixture(driver_index: int, fixture: str, count: int) -> list[float]:
    if fixture == "ZERO_SOURCE":
        return [0.0] * count
    if fixture == "SMOOTH_BOUNDED_SOURCE":
        scale = 0.08 + 0.012 * driver_index
        return [
            math.tanh(scale + 0.45 * x + 0.08 * math.sin((driver_index + 1) * math.pi * x))
            for x in (index / (count - 1) for index in range(count))
        ]
    if fixture == "STEP_BOUNDED_SOURCE":
        low = min(0.45, 0.05 + 0.01 * driver_index)
        high = min(0.95, low + 0.35)
        return [low if index < count // 2 else high for index in range(count)]
    raise TwellCompilerError(f"unknown synthetic fixture: {fixture}")


def _combine(
    compound_id: str, first: list[float], second: list[float]
) -> tuple[list[float], dict[str, list[float]]]:
    combined: list[float] = []
    for u1, u2 in zip(first, second, strict=True):
        if compound_id == "X01":
            value = max(-1.0, min(1.0, u1 * (1 + u2) / 2))
        elif compound_id in {"X02", "X06", "X16"}:
            value = u1 * u2
        elif compound_id == "X03":
            value = u1 * (1 - abs(u2))
        elif compound_id == "X04":
            value = (u1 + u2 + u1 * u2) / 3
        elif compound_id == "X05":
            value = (u1 - u2) / 2
        elif compound_id == "X07":
            value = math.copysign(math.sqrt(abs(u1 * u2)), u1 * u2) if u1 * u2 else 0.0
        elif compound_id == "X08":
            value = u1 * u2 / (1 + abs(u1 * u2))
        elif compound_id == "X09":
            value = (2 * u1 + u2) / 3
        elif compound_id in {"X10", "X17"}:
            value = u1 / (1 + abs(u2))
        elif compound_id == "X11":
            value = math.tanh(u1 + u2)
        elif compound_id == "X12":
            value = u1 * math.cos(math.pi * u2)
        elif compound_id == "X13":
            value = (u1 + 2 * u2) / 3
        elif compound_id == "X14":
            value = u1 * (1 + u2) / 2
        elif compound_id == "X15":
            value = u2 / (1 + abs(u1))
        elif compound_id == "X18":
            value = (u1 + u2) / 2
        elif compound_id == "X19":
            value = u1
        elif compound_id == "X20":
            value = u2
        else:
            raise TwellCompilerError(f"unknown frozen compound operation: {compound_id}")
        combined.append(value)
    return combined, {"u1": list(first), "u2": list(second)}


def _parameter_cells(
    concept_id: str,
    architecture: Mapping[str, Any],
    compound: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Enumerate every frozen architecture parameter tuple plus two explicit overrides."""

    parameters = list(architecture["parameters"])
    names = [str(row["name"]) for row in parameters]
    value_axes = [list(row["values"]) for row in parameters]
    cells = [
        {
            "cell_id": f"{concept_id}-C{index:03d}",
            "cell_kind": "CARTESIAN",
            "parameters": dict(zip(names, values, strict=True)),
        }
        for index, values in enumerate(itertools.product(*value_axes), start=1)
    ]
    compound_id = None if compound is None else str(compound["id"])
    if compound_id in {"X19", "X20"}:
        evidence_parameters = {str(row["name"]): max(row["values"]) for row in parameters}
        cells.append(
            {
                "cell_id": f"{concept_id}-C{len(cells) + 1:03d}",
                "cell_kind": "COMPOUND_OVERRIDE_EVIDENCE",
                "parameters": evidence_parameters,
            }
        )
    return cells


def _solve_tridiagonal(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> list[float]:
    """Solve a nonsingular tridiagonal system using the Thomas algorithm."""

    count = len(diagonal)
    if not (len(lower) == len(upper) == count - 1 and len(rhs) == count):
        raise TwellCompilerError("invalid tridiagonal system dimensions")
    c_prime = [0.0] * max(0, count - 1)
    d_prime = [0.0] * count
    pivot = float(diagonal[0])
    if abs(pivot) <= 1e-15:
        raise TwellCompilerError("singular tridiagonal system at row 0")
    if count > 1:
        c_prime[0] = float(upper[0]) / pivot
    d_prime[0] = float(rhs[0]) / pivot
    for index in range(1, count):
        pivot = float(diagonal[index]) - float(lower[index - 1]) * c_prime[index - 1]
        if abs(pivot) <= 1e-15:
            raise TwellCompilerError(f"singular tridiagonal system at row {index}")
        if index < count - 1:
            c_prime[index] = float(upper[index]) / pivot
        d_prime[index] = (float(rhs[index]) - float(lower[index - 1]) * d_prime[index - 1]) / pivot
    result = [0.0] * count
    result[-1] = d_prime[-1]
    for index in range(count - 2, -1, -1):
        result[index] = d_prime[index] - c_prime[index] * result[index + 1]
    return result


def _neumann_helmholtz(values: Sequence[float], ell: float) -> tuple[list[float], dict[str, float]]:
    """Solve q-ell^2 q''=u with the exact frozen first-difference rows."""

    count = len(values)
    dx = 1.0 / (count - 1)
    ratio = ell * ell / (dx * dx)
    lower = [-ratio] * (count - 1)
    diagonal = [1.0 + 2.0 * ratio] * count
    upper = [-ratio] * (count - 1)
    rhs = [float(value) for value in values]
    diagonal[0] = -1.0
    upper[0] = 1.0
    rhs[0] = 0.0
    lower[-1] = -1.0
    diagonal[-1] = 1.0
    rhs[-1] = 0.0
    state = _solve_tridiagonal(lower, diagonal, upper, rhs)
    operator_residual = max(
        abs(
            -ratio * state[index - 1]
            + (1.0 + 2.0 * ratio) * state[index]
            - ratio * state[index + 1]
            - float(values[index])
        )
        for index in range(1, count - 1)
    )
    boundary_residual = max(abs(state[1] - state[0]), abs(state[-1] - state[-2]))
    return state, {
        "operator_residual": operator_residual,
        "boundary_residual": boundary_residual,
    }


def _mixed_boundary_helmholtz(
    values: Sequence[float], mu: float
) -> tuple[list[float], dict[str, float]]:
    """Solve q''-mu^2 q=-u with q'(0)=0 and q(1)=0."""

    count = len(values)
    dx = 1.0 / (count - 1)
    inverse_dx2 = 1.0 / (dx * dx)
    lower = [inverse_dx2] * (count - 1)
    diagonal = [-(2.0 * inverse_dx2 + mu * mu)] * count
    upper = [inverse_dx2] * (count - 1)
    rhs = [-float(value) for value in values]
    diagonal[0] = -1.0
    upper[0] = 1.0
    rhs[0] = 0.0
    lower[-1] = 0.0
    diagonal[-1] = 1.0
    rhs[-1] = 0.0
    state = _solve_tridiagonal(lower, diagonal, upper, rhs)
    operator_residual = max(
        abs(
            inverse_dx2 * state[index - 1]
            - (2.0 * inverse_dx2 + mu * mu) * state[index]
            + inverse_dx2 * state[index + 1]
            + float(values[index])
        )
        for index in range(1, count - 1)
    )
    boundary_residual = max(abs(state[1] - state[0]), abs(state[-1]))
    return state, {
        "operator_residual": operator_residual,
        "boundary_residual": boundary_residual,
    }


def _dimension_evidence(
    drivers: Sequence[Mapping[str, Any]],
    architecture: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the declared dimension chain rather than copying a PASS label."""

    residuals: list[str] = []
    normalization_proofs: list[dict[str, str]] = []
    for driver in drivers:
        expression = str(driver["normalized_expression"])
        dimension = str(driver["dimension"])
        if not expression.startswith("u=tanh(") or not expression.endswith(")"):
            residuals.append(f"{driver['id']}:missing_normalization")
        if dimension not in {"1", "dimensionless"} and "_ref" not in expression:
            residuals.append(f"{driver['id']}:missing_equal_dimension_reference")
        normalization_proofs.append(
            {
                "driver_id": str(driver["id"]),
                "source_dimension": dimension,
                "reference_unit": str(driver["reference_unit"]),
                "normalized_expression": expression,
                "normalized_output_dimension": "1",
            }
        )
    parameter_units = {str(row["name"]): str(row["unit"]) for row in architecture["parameters"]}
    if set(parameters) != set(parameter_units):
        residuals.append("parameter_set_mismatch")
    if parameter_units.get("lambda") != "1":
        residuals.append("lambda_not_dimensionless")
    for name, value in parameters.items():
        if name not in parameter_units or not math.isfinite(float(value)):
            residuals.append(f"{name}:invalid_parameter")
    signature = {
        "normalized_driver_unit": "1",
        "response_A_unit": "1",
        "baryonic_acceleration_unit": "L*T^-2",
        "effective_acceleration_unit": "L*T^-2",
        "parameter_units": parameter_units,
        "architecture": architecture["id"],
        "normalization_proofs": normalization_proofs,
    }
    return {
        "computed": True,
        "status": "PASS" if not residuals else "FAIL",
        "residual_count": len(residuals),
        "residuals": residuals,
        "signature_sha256": content_sha256(signature),
        "signature": signature,
    }


def _evaluate_operator(
    architecture_id: str,
    values: Sequence[float],
    parameters: Mapping[str, Any],
    seed: int,
    context: Mapping[str, Sequence[float]] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Execute the exact frozen operator for one parameter cell and fixture."""

    count = len(values)
    x = [index / (count - 1) for index in range(count)]
    dx = 1.0 / (count - 1)
    lam = float(parameters["lambda"])
    context = context or {}
    state: list[float] | None = None
    operator_residual = 0.0
    boundary_residual = 0.0
    effective_parameters = {key: value for key, value in parameters.items()}
    if architecture_id in {"A01_LAPSE", "A02_CLOCK", "A08_PERMITTIVITY"}:
        output = [math.exp(lam * value) for value in values]
    elif architecture_id == "A03_CONFORMAL":
        output = [1.0 + lam * value for value in values]
    elif architecture_id == "A04_DISFORMAL":
        output = [1.0 + lam * value * value / (1.0 + value * value) for value in values]
    elif architecture_id == "A05_SLIP":
        output = [1.0 + 0.5 * lam * value for value in values]
    elif architecture_id == "A06_SPATIAL_KERNEL":
        state, residuals = _neumann_helmholtz(values, float(parameters["ell"]))
        operator_residual = residuals["operator_residual"]
        boundary_residual = residuals["boundary_residual"]
        output = [math.exp(lam * value) for value in state]
    elif architecture_id == "A07_BOUNDARY":
        state = [0.0] * count
        for index in range(count - 2, -1, -1):
            state[index] = state[index + 1] + 0.5 * dx * (
                float(values[index]) + float(values[index + 1])
            )
        operator_residual = max(
            abs(
                state[index]
                - state[index + 1]
                - 0.5 * dx * (float(values[index]) + float(values[index + 1]))
            )
            for index in range(count - 1)
        )
        boundary_residual = abs(state[-1])
        output = [math.exp(lam * value) for value in state]
    elif architecture_id == "A09_ENTROPIC":
        output = [1.0 + lam * value / (1.0 + abs(value)) for value in values]
    elif architecture_id == "A10_DENSITY_SCREEN":
        critical = float(parameters["u_c"])
        power = float(parameters["n"])
        output = [1.0 + lam / (1.0 + (abs(value) / critical) ** power) for value in values]
    elif architecture_id == "A11_DERIV_SCREEN":
        derivative: list[float] = []
        for index in range(count):
            left = values[max(0, index - 1)]
            right = values[min(count - 1, index + 1)]
            width = dx if index in {0, count - 1} else 2 * dx
            derivative.append(abs(right - left) / width)
        critical = float(parameters["s_c"])
        power = float(parameters["n"])
        output = [1.0 + lam / (1.0 + (slope / critical) ** power) for slope in derivative]
    elif architecture_id == "A12_MASSIVE":
        state, residuals = _mixed_boundary_helmholtz(values, float(parameters["mu"]))
        operator_residual = residuals["operator_residual"]
        boundary_residual = residuals["boundary_residual"]
        output = [1.0 + lam * value for value in state]
    elif architecture_id == "A13_MIXED_MODE":
        nonlocal_state, residuals = _neumann_helmholtz(values, float(parameters["ell"]))
        operator_residual = residuals["operator_residual"]
        boundary_residual = residuals["boundary_residual"]
        theta = float(parameters["theta"])
        state = [
            math.cos(theta) * float(local) + math.sin(theta) * nonlocal_value
            for local, nonlocal_value in zip(values, nonlocal_state, strict=True)
        ]
        output = [1.0 + lam * value for value in state]
    elif architecture_id == "A14_PHASE":
        wave_number = float(parameters["k"])
        phase_zero = float(parameters["phi0"])
        output = [
            1.0 + lam * value * math.cos(2.0 * math.pi * wave_number * xi + phase_zero)
            for value, xi in zip(values, x, strict=True)
        ]
    elif architecture_id == "A15_RETARDED":
        if float(parameters["c_g"]) != 1.0:
            raise TwellCompilerError("A15 exact template only defines c_g=1")
        state = [0.0, *[float(value) for value in values[:-1]]]
        boundary_residual = abs(state[0])
        operator_residual = max(
            abs(state[index] - float(values[index - 1])) for index in range(1, count)
        )
        output = [math.exp(lam * value) for value in state]
    elif architecture_id == "A16_MEMORY":
        tau = float(parameters["tau"])
        decay = math.exp(-dx / tau)
        state = [0.0]
        for index in range(1, count):
            state.append(float(values[index]) + (state[index - 1] - float(values[index])) * decay)
        boundary_residual = abs(state[0])
        operator_residual = max(
            abs(
                state[index]
                - float(values[index])
                - (state[index - 1] - float(values[index])) * decay
            )
            for index in range(1, count)
        )
        output = [math.exp(lam * value) for value in state]
    elif architecture_id == "A17_RESONANCE":
        omega = float(parameters["omega"])
        damping = float(parameters["zeta"])
        state = [0.0]
        velocities = [0.0]
        recurrence_residuals: list[float] = []
        for index in range(1, count):
            expected_velocity = velocities[index - 1] + dx * (
                omega * omega * (float(values[index]) - state[index - 1])
                - 2.0 * damping * omega * velocities[index - 1]
            )
            expected_state = state[index - 1] + dx * expected_velocity
            velocities.append(expected_velocity)
            state.append(expected_state)
            recurrence_residuals.extend(
                [
                    abs(velocities[index] - expected_velocity),
                    abs(state[index] - expected_state),
                ]
            )
        operator_residual = max(recurrence_residuals, default=0.0)
        boundary_residual = max(abs(state[0]), abs(velocities[0]))
        output = [1.0 + lam * value for value in state]
    elif architecture_id == "A18_STOCHASTIC":
        tau = float(parameters["tau"])
        sigma_cell = float(parameters["sigma"])
        u2 = context.get("u2")
        generator = random.Random(seed)
        draws = [0.0, *[generator.gauss(0.0, 1.0) for _ in range(1, count)]]
        state = [0.0]
        recurrence_residuals = []
        effective_sigma: list[float] = [0.0]
        for index in range(1, count):
            sigma_eff = sigma_cell * (abs(float(u2[index])) if u2 is not None else 1.0)
            effective_sigma.append(sigma_eff)
            expected = (
                state[index - 1]
                + dx * (float(values[index]) - state[index - 1]) / tau
                + sigma_eff * math.sqrt(dx) * draws[index]
            )
            state.append(expected)
            recurrence_residuals.append(abs(state[index] - expected))
        effective_parameters["sigma_eff_digest_sha256"] = content_sha256(effective_sigma)
        operator_residual = max(recurrence_residuals, default=0.0)
        boundary_residual = abs(state[0])
        output = [math.exp(lam * value) for value in state]
    elif architecture_id == "A19_FEEDBACK":
        kappa_cell = float(parameters["kappa"])
        u1 = context.get("u1")
        state = []
        fixed_point_residuals: list[float] = []
        effective_kappa: list[float] = []
        for index, value in enumerate(values):
            kappa_eff = kappa_cell * (abs(float(u1[index])) if u1 is not None else 1.0)
            effective_kappa.append(kappa_eff)
            q = 0.0
            for _ in range(128):
                updated = math.tanh(float(value) + kappa_eff * q)
                if abs(updated - q) <= 1e-12:
                    q = updated
                    break
                q = updated
            else:
                raise TwellCompilerError("feedback probe did not converge")
            state.append(q)
            fixed_point_residuals.append(abs(q - math.tanh(float(value) + kappa_eff * q)))
        effective_parameters["kappa_eff_digest_sha256"] = content_sha256(effective_kappa)
        operator_residual = max(fixed_point_residuals, default=0.0)
        output = [math.exp(lam * value) for value in state]
    else:
        raise TwellCompilerError(f"unknown architecture probe: {architecture_id}")

    finite_residual = sum(0 if math.isfinite(value) else 1 for value in output)
    return output, {
        "computed": True,
        "operator_residual": operator_residual,
        "boundary_or_initial_residual": boundary_residual,
        "finite_violation_count": finite_residual,
        "effective_parameters": effective_parameters,
        "state_digest_sha256": content_sha256(state) if state is not None else None,
    }


def _probe_entry(
    entry_id: str,
    drivers: Sequence[Mapping[str, Any]],
    architecture: Mapping[str, Any],
    compound: Mapping[str, Any] | None,
    driver_indexes: Mapping[str, int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    point_count = int(config["probe_contract"]["fixture_grid_points"])
    seed = int(config["compiler_contract"]["deterministic_seed"])
    operator_tolerance = float(config["probe_contract"]["operator_residual_tolerance"])
    boundary_tolerance = float(config["probe_contract"]["boundary_residual_tolerance"])
    null_tolerance = float(config["probe_contract"]["null_limit_tolerance"])
    driver_ids = [str(row["id"]) for row in drivers]
    cells = _parameter_cells(entry_id, architecture, compound)
    cell_results: list[dict[str, Any]] = []
    aggregate_operator_residual = 0.0
    aggregate_boundary_residual = 0.0
    aggregate_analytic_residual = 0.0
    for cell in cells:
        parameters = dict(cell["parameters"])
        dimension = _dimension_evidence(drivers, architecture, parameters)
        fixture_results: list[dict[str, Any]] = []
        failures: list[str] = []
        for fixture_name in config["probe_contract"]["fixtures"]:
            first = _fixture(driver_indexes[driver_ids[0]], fixture_name, point_count)
            context: dict[str, Sequence[float]] = {}
            if len(driver_ids) == 2:
                second = _fixture(driver_indexes[driver_ids[1]], fixture_name, point_count)
                first, context = _combine(entry_id, first, second)
            try:
                output, evidence = _evaluate_operator(
                    str(architecture["id"]), first, parameters, seed, context
                )
                replay, replay_evidence = _evaluate_operator(
                    str(architecture["id"]), first, parameters, seed, context
                )
                null_parameters = dict(parameters)
                null_parameters["lambda"] = 0.0
                null, null_evidence = _evaluate_operator(
                    str(architecture["id"]), first, null_parameters, seed, context
                )
                deterministic_residual = 0.0 if output == replay else math.inf
                deterministic_evidence_residual = 0.0 if evidence == replay_evidence else math.inf
                finite_violation_count = int(evidence["finite_violation_count"])
                analytic_residual = max(abs(value - 1.0) for value in null)
                operator_residual = max(
                    float(evidence["operator_residual"]),
                    float(null_evidence["operator_residual"]),
                )
                boundary_residual = max(
                    float(evidence["boundary_or_initial_residual"]),
                    float(null_evidence["boundary_or_initial_residual"]),
                )
                if deterministic_residual or deterministic_evidence_residual:
                    failures.append(f"{fixture_name}:DETERMINISTIC_REPLAY")
                if finite_violation_count:
                    failures.append(f"{fixture_name}:FINITE")
                if operator_residual > operator_tolerance:
                    failures.append(f"{fixture_name}:COMPUTED_OPERATOR_RESIDUAL")
                if boundary_residual > boundary_tolerance:
                    failures.append(f"{fixture_name}:COMPUTED_BOUNDARY_OR_INITIAL_CONDITION")
                if analytic_residual > null_tolerance:
                    failures.append(f"{fixture_name}:COMPUTED_ANALYTIC_LIMIT")
                aggregate_operator_residual = max(aggregate_operator_residual, operator_residual)
                aggregate_boundary_residual = max(aggregate_boundary_residual, boundary_residual)
                aggregate_analytic_residual = max(aggregate_analytic_residual, analytic_residual)
                fixture_results.append(
                    {
                        "fixture": fixture_name,
                        "output_sha256": content_sha256(output),
                        "output_min": min(output),
                        "output_max": max(output),
                        "computed_operator_residual": operator_residual,
                        "computed_boundary_or_initial_residual": boundary_residual,
                        "computed_analytic_lambda_zero_residual": analytic_residual,
                        "finite_violation_count": finite_violation_count,
                        "deterministic_replay_residual": deterministic_residual,
                        "deterministic_evidence_replay_residual": (deterministic_evidence_residual),
                        "operator_evidence": evidence,
                    }
                )
            except (ArithmeticError, TwellCompilerError, ValueError) as error:
                failures.append(f"{fixture_name}:EXECUTION:{type(error).__name__}:{error}")
                fixture_results.append(
                    {
                        "fixture": fixture_name,
                        "execution_error": f"{type(error).__name__}:{error}",
                    }
                )
        if dimension["status"] != "PASS":
            failures.append("COMPUTED_DIMENSION")
        status = (
            "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES" if not failures else "INCOMPLETE_QUARANTINE"
        )
        cell_results.append(
            {
                **cell,
                "status": status,
                "execution_class": (
                    "EXACT_FROZEN_OPERATOR" if not failures else "FORMULA_BASIS_ONLY"
                ),
                "computed_dimension_evidence": dimension,
                "fixture_results": fixture_results,
                "failures": failures,
                "cell_evidence_sha256": content_sha256(
                    {
                        "parameters": parameters,
                        "dimension": dimension,
                        "fixtures": fixture_results,
                        "failures": failures,
                    }
                ),
            }
        )
    passed_count = sum(
        row["status"] == "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES" for row in cell_results
    )
    entry_status = (
        "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES"
        if passed_count == len(cell_results)
        else "INCOMPLETE_QUARANTINE"
    )
    behavior_projection = [
        {
            "cell_kind": row["cell_kind"],
            "parameters": row["parameters"],
            "status": row["status"],
            "dimension_status": row["computed_dimension_evidence"]["status"],
            "fixtures": [
                {
                    key: fixture[key]
                    for key in (
                        "fixture",
                        "output_sha256",
                        "output_min",
                        "output_max",
                        "computed_operator_residual",
                        "computed_boundary_or_initial_residual",
                        "computed_analytic_lambda_zero_residual",
                        "finite_violation_count",
                        "deterministic_replay_residual",
                        "deterministic_evidence_replay_residual",
                        "execution_error",
                    )
                    if key in fixture
                }
                for fixture in row["fixture_results"]
            ],
        }
        for row in cell_results
    ]
    return {
        "status": entry_status,
        "execution_class": (
            "EXACT_FROZEN_OPERATOR"
            if entry_status == "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES"
            else "FORMULA_BASIS_ONLY"
        ),
        "checks": list(config["probe_contract"]["required_checks"]),
        "parameter_cell_count": len(cell_results),
        "passed_parameter_cell_count": passed_count,
        "failed_parameter_cell_count": len(cell_results) - passed_count,
        "maximum_computed_operator_residual": aggregate_operator_residual,
        "maximum_computed_boundary_or_initial_residual": aggregate_boundary_residual,
        "maximum_computed_analytic_lambda_zero_residual": aggregate_analytic_residual,
        "fixture_digest_sha256": content_sha256(behavior_projection),
        "cell_results": cell_results,
    }


def _source_status(
    drivers: Sequence[Mapping[str, Any]], architecture_id: str, domain: str, override: Any
) -> tuple[str, str]:
    if architecture_id in {"A15_RETARDED", "A16_MEMORY"}:
        return "SOURCE_BLOCKED_ARCHITECTURE_REQUIRES_HISTORY", "SOURCE_BLOCKED"
    if override is not None:
        status = str(override)
    else:
        key = "sparc_source_status" if domain == "SPARC" else "xcop_source_status"
        statuses = [str(driver[key]) for driver in drivers]
        status = statuses[0] if len(statuses) == 1 else " + ".join(statuses)
        if not all(item.startswith(SOURCE_READY_PREFIX) for item in statuses):
            return status, "SOURCE_BLOCKED"
    admission = (
        "READY_FOR_THEORY_GATES" if status.startswith(SOURCE_READY_PREFIX) else "SOURCE_BLOCKED"
    )
    return status, admission


def _card(
    concept_id: str,
    drivers: Sequence[Mapping[str, Any]],
    architecture: Mapping[str, Any],
    compound: Mapping[str, Any] | None,
    probe: Mapping[str, Any],
    hashes: Mapping[str, str],
    prior_card_sha256: str,
) -> dict[str, Any]:
    driver_ids = [str(row["id"]) for row in drivers]
    formula_basis = {
        "drivers": [
            {
                "id": row["id"],
                "normalization": row["normalized_expression"],
                "dimension": row["dimension"],
            }
            for row in drivers
        ],
        "architecture": architecture["id"],
        "expressions": architecture["canonical_expressions"],
        "compound_operation": None if compound is None else compound["operation"],
    }
    formula_sha = content_sha256(formula_basis)
    limits = {
        "initial_conditions": architecture["initial_conditions"],
        "boundaries": architecture["boundaries"],
        "null": "lambda=0 gives A=1 and g_eff=g_b",
    }
    observable_fingerprint = content_sha256(
        {
            "probe": probe["fixture_digest_sha256"],
            "matter": architecture["matter_closure"],
            "photon": "L0_NO_LIGHT_CLAIM",
            "capture": "C0_ISOLATED_CONSERVATIVE",
            "gw": architecture["gw_closure"],
        }
    )
    operational_variables = [
        {
            "symbol": row["symbol"],
            "operational_definition": row["source_definition"],
            "dimension": row["dimension"],
            "observable_or_latent": "SOURCE_DERIVED",
        }
        for row in drivers
    ]
    operational_variables.extend(
        [
            {
                "symbol": "u_D",
                "operational_definition": "frozen bounded driver or compound normalization",
                "dimension": "1",
                "observable_or_latent": "SOURCE_DERIVED",
            },
            {
                "symbol": "A",
                "operational_definition": "dimensionless effective matter-response multiplier",
                "dimension": "1",
                "observable_or_latent": "LATENT_FIELD",
            },
        ]
    )
    parameter_units = {
        str(parameter["name"]): str(parameter["unit"]) for parameter in architecture["parameters"]
    }
    parameters = [
        {
            "cell_id": row["cell_id"],
            "parameter": row["cell_kind"],
            "value": dict(row["parameters"]),
            "unit": parameter_units,
            "frozen": True,
        }
        for row in probe["cell_results"]
    ]
    state_mode = architecture["state_mode"]
    quantum = "CLASSICAL_LIMIT_ONLY" if "QG03" in architecture["qg_nodes"] else "NOT_APPLICABLE"
    fields = (
        ["A_effective"]
        if architecture["kind"] == "EFFECTIVE_LAW"
        else ["q_effective", "A_effective"]
    )
    source_text = "; ".join(f"{row['id']}:{row['source_definition']}" for row in drivers)
    if compound is not None:
        source_text += f"; compound={compound['operation']}"
    return {
        "schema_version": CARD_SCHEMA,
        "card_id": f"{concept_id}@2.1.0",
        "stable_concept_id": concept_id,
        "semantic_version": "2.1.0",
        "identity_class": "FORMULA_VARIANT" if compound is None else "NEW_CONCEPT",
        "parents": [],
        "author_agent": "target-blind-twell-400-v2-typed-compiler-v1",
        "provenance": {
            "created_at_utc": "2026-08-30T00:00:00Z",
            "origin_timing": "PRE_RESPONSE",
            "origin_artifacts": [PACKET_ID, "TWELL-400-v2"],
            "residual_access_lineage": [],
        },
        "lay_mechanism": (
            f"The {architecture['name']} template lets source-side "
            f"{', '.join(row['name'] for row in drivers)} alter an effective matter field."
        ),
        "novelty_claim": (
            "Formula-basis registration only; no historical novelty is inferred from this "
            "driver-architecture composition."
        ),
        "ontology": ["TWELL-400-v2", *architecture["qg_nodes"]],
        "scientific_status": "H_HYPOTHESIS",
        "operational_variables": operational_variables,
        "source": source_text,
        "coupling": "universal source-side scalar effective coupling to radial matter dynamics",
        "action_or_equations": {
            "kind": architecture["kind"],
            "exact_expressions": [
                *architecture["canonical_expressions"],
                "compound=" + ("NONE" if compound is None else compound["operation"]),
            ],
            "executable": probe["execution_class"] == "EXACT_FROZEN_OPERATOR",
        },
        "initial_conditions": list(architecture["initial_conditions"]),
        "boundaries": list(architecture["boundaries"]),
        "degrees_of_freedom": {
            "fields": fields,
            "spin_helicity": "classical scalar phenomenology; no fundamental spin assignment",
            "mass": "effective template-specific; not a particle-mass claim",
            "statistics": "not applicable",
            "state": "seeded stochastic"
            if state_mode == "STOCHASTIC"
            else "classical deterministic",
            "quantum_applicability": quantum,
        },
        "propagation": {
            "speed": "c cell only"
            if state_mode == "RETARDED"
            else "no independently claimed propagation speed",
            "dispersion": "not derived",
            "polarization": "none assigned to scalar phenomenology",
            "attenuation": "template-defined only",
            "range": "parameter-grid or normalized-domain range only",
            "static_limit": "lambda=0 recovers the baryonic matter law",
        },
        "state_rule": {
            "mode": state_mode,
            "exact_rule": "; ".join(architecture["canonical_expressions"]),
        },
        "closures": {
            "matter": architecture["matter_closure"],
            "photon": "L0_NO_LIGHT_CLAIM",
            "gravitational_wave": architecture["gw_closure"],
            "quantum_laboratory": "Q0_NO_QUANTUM_LAB_CLAIM",
            "capture": "C0_ISOLATED_CONSERVATIVE",
            "cosmology": "COS0_NO_COSMOLOGY_CLAIM",
        },
        "ledgers": {
            "energy": "phenomenology only; action and conserved energy required before promotion",
            "momentum": "no momentum exchange claimed; theory gate must derive it",
            "entropy": "no entropy production claimed",
            "information": "source-only construction; no response information enters the state",
        },
        "structure": {
            "symmetries": [
                "one-dimensional target-free synthetic domain",
                "radial phenomenology when applied radially",
            ],
            "covariance_or_frame": "declared source hypersurface; no covariant completion claimed",
            "equivalence_behavior": "universal matter multiplier in the effective radial closure only",
            "causal_structure": "retarded only where explicitly stated; otherwise no causal completion claimed",
        },
        "dimensions": [
            *[f"[{row['symbol']}]={row['dimension']}" for row in drivers],
            "[u_D]=[A]=1",
            "[g_eff]=[g_b]=L T^-2",
        ],
        "parameter_cells": parameters,
        "priors": ["finite frozen grid only; no fit, posterior, or score in compiler"],
        "screens": ["null lambda=0 cell", "finite/limit/deterministic target-free probes"],
        "limiting_cases": [
            "lambda=0 gives A=1 and g_eff=g_b",
            *architecture["boundaries"],
        ],
        "source_only_data_contract": {
            "allowed_inputs": [*driver_ids, *architecture["source_needs"]],
            "forbidden_response_inputs": FORBIDDEN_RESPONSE_INPUTS,
            "construction_before_response": True,
            "missing_data_action": "SOURCE_BLOCKED",
        },
        "synthetic_falsifier": (
            "Reject or quarantine if schema, dimensions, finiteness, deterministic replay, "
            "boundary/initial conditions, or lambda=0 baryonic limit fails on any frozen fixture."
        ),
        "real_data_discriminator": (
            "Future target-blind campaign manifest only; this compiler opens no response and "
            "issues no DATA_ELIGIBLE label."
        ),
        "prior_art": [
            {
                "citation": source_id,
                "relationship": "bounded comparator or ontology anchor; no novelty inference",
            }
            for source_id in architecture["prior_art_source_ids"]
        ],
        "equivalence_fingerprint": {
            "canonical_symbolic_sha256": formula_sha,
            "analytic_limits_sha256": content_sha256(limits),
            "synthetic_fingerprint_sha256": str(probe["fixture_digest_sha256"]),
            "observable_fingerprint_sha256": observable_fingerprint,
        },
        "version_change": {
            "kind": "MINOR",
            "previous_card_id": f"{concept_id}@2.0.0",
            "previous_card_sha256": prior_card_sha256,
            "changed_facets": [
                "action_or_equations",
                "parameter_cells",
                "synthetic_falsifier",
                "equivalence_fingerprint",
            ],
            "prior_result_retained": True,
            "replay_all_affected": True,
        },
        "hashes": {
            "code_sha256": hashes["code_sha256"],
            "data_sha256": hashes["data_sha256"],
            "environment_sha256": hashes["environment_sha256"],
            "configuration_sha256": hashes["configuration_sha256"],
            "formula_sha256": formula_sha,
        },
    }


CARD_REQUIRED_KEYS = {
    "schema_version",
    "card_id",
    "stable_concept_id",
    "semantic_version",
    "identity_class",
    "parents",
    "author_agent",
    "provenance",
    "lay_mechanism",
    "novelty_claim",
    "ontology",
    "scientific_status",
    "operational_variables",
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
    "source_only_data_contract",
    "synthetic_falsifier",
    "real_data_discriminator",
    "prior_art",
    "equivalence_fingerprint",
    "version_change",
    "hashes",
}


def _internal_card_errors(card: Mapping[str, Any]) -> list[str]:
    """Validate the frozen internal card contract while the central schema is deferred."""

    errors: list[str] = []
    if set(card) != CARD_REQUIRED_KEYS:
        errors.append("top_level_keys")
    concept_id = str(card.get("stable_concept_id", ""))
    if concept_id not in set(ordered_concept_ids()):
        errors.append("stable_concept_id")
    if card.get("card_id") != f"{concept_id}@2.1.0":
        errors.append("card_id")
    if card.get("semantic_version") != "2.1.0":
        errors.append("semantic_version")
    action = card.get("action_or_equations")
    if not isinstance(action, Mapping) or set(action) != {
        "kind",
        "exact_expressions",
        "executable",
    }:
        errors.append("action_or_equations")
    elif not isinstance(action["executable"], bool) or not action["exact_expressions"]:
        errors.append("action_or_equations_values")
    closures = card.get("closures")
    if not isinstance(closures, Mapping):
        errors.append("closures")
    elif (
        closures.get("photon") != "L0_NO_LIGHT_CLAIM"
        or closures.get("capture") != "C0_ISOLATED_CONSERVATIVE"
    ):
        errors.append("closure_defaults")
    parameter_cells = card.get("parameter_cells")
    if not isinstance(parameter_cells, list) or not parameter_cells:
        errors.append("parameter_cells")
    else:
        cell_ids = [row.get("cell_id") for row in parameter_cells if isinstance(row, Mapping)]
        if len(cell_ids) != len(parameter_cells) or len(cell_ids) != len(set(cell_ids)):
            errors.append("parameter_cell_ids")
        for row in parameter_cells:
            if not isinstance(row, Mapping) or set(row) != {
                "cell_id",
                "parameter",
                "value",
                "unit",
                "frozen",
            }:
                errors.append("parameter_cell_shape")
                break
            if row["frozen"] is not True or not isinstance(row["value"], Mapping):
                errors.append("parameter_cell_value")
                break
    hashes = card.get("hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != {
        "code_sha256",
        "data_sha256",
        "environment_sha256",
        "configuration_sha256",
        "formula_sha256",
    }:
        errors.append("hashes")
    elif not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value) for value in hashes.values()
    ):
        errors.append("hash_values")
    fingerprint = card.get("equivalence_fingerprint")
    if not isinstance(fingerprint, Mapping) or not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value) for value in fingerprint.values()
    ):
        errors.append("equivalence_fingerprint")
    version = card.get("version_change")
    if not isinstance(version, Mapping) or (
        version.get("kind") != "MINOR"
        or version.get("previous_card_id") != f"{concept_id}@2.0.0"
        or not isinstance(version.get("previous_card_sha256"), str)
        or SHA256_RE.fullmatch(str(version.get("previous_card_sha256"))) is None
        or version.get("prior_result_retained") is not True
        or version.get("replay_all_affected") is not True
    ):
        errors.append("version_change")
    data_contract = card.get("source_only_data_contract")
    if not isinstance(data_contract, Mapping) or (
        data_contract.get("forbidden_response_inputs") != FORBIDDEN_RESPONSE_INPUTS
        or data_contract.get("construction_before_response") is not True
        or data_contract.get("missing_data_action") != "SOURCE_BLOCKED"
    ):
        errors.append("source_only_data_contract")
    return errors


def _failed_card_hashes(payload: bytes) -> dict[str, str]:
    """Verify and return the append-only lineage of the retained blocked v2.0 cards."""

    rows: list[dict[str, Any]] = []
    try:
        for line in payload.splitlines():
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("wrapper must be an object")
                rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise TwellCompilerError("failed-card counterevidence is not valid JSONL") from error
    if [row.get("concept_id") for row in rows] != ordered_concept_ids():
        raise TwellCompilerError("failed-card counterevidence ordering changed")
    lineage: dict[str, str] = {}
    for row in rows:
        concept_id = str(row["concept_id"])
        card = row.get("card")
        if not isinstance(card, Mapping) or card.get("card_id") != f"{concept_id}@2.0.0":
            raise TwellCompilerError(f"failed-card lineage changed: {concept_id}")
        computed = content_sha256(card)
        if row.get("card_sha256") != computed:
            raise TwellCompilerError(f"failed-card self-hash changed: {concept_id}")
        lineage[concept_id] = computed
    return lineage


def compile_rows(
    config: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    prior_card_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    drivers = {row["id"]: row for row in config["driver_catalog"]}
    architectures = {row["id"]: row for row in config["architecture_catalog"]}
    driver_indexes = {driver_id: index for index, driver_id in enumerate(drivers, start=1)}
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
    for compound in config["compound_catalog"]:
        entries.append(
            (compound["id"], list(compound["drivers"]), compound["architecture"], compound)
        )
    if [entry[0] for entry in entries] != ordered_concept_ids():
        raise TwellCompilerError("compiled concept ordering changed")

    compiled: list[dict[str, Any]] = []
    for order_index, (concept_id, driver_ids, architecture_id, compound) in enumerate(entries):
        driver_rows = [drivers[driver_id] for driver_id in driver_ids]
        architecture = architectures[architecture_id]
        probe = _probe_entry(
            concept_id, driver_rows, architecture, compound, driver_indexes, config
        )
        card = _card(
            concept_id,
            driver_rows,
            architecture,
            compound,
            probe,
            artifact_hashes,
            prior_card_hashes[concept_id],
        )
        errors = _internal_card_errors(card)
        if errors:
            raise TwellCompilerError(
                f"internal typed card contract failed: {concept_id}: {errors[:3]}"
            )
        domain_admission: dict[str, str] = {}
        source_snapshot: dict[str, str] = {}
        for domain, override_key in (("SPARC", "sparc_override"), ("XCOP", "xcop_override")):
            override = None if compound is None else compound[override_key]
            status, admission = _source_status(driver_rows, architecture_id, domain, override)
            source_snapshot[domain] = status
            domain_admission[domain] = admission
        if probe["status"] != "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES":
            domain_admission = {domain: "INCOMPLETE_QUARANTINE" for domain in domain_admission}
            compiler_status = "INCOMPLETE_QUARANTINE"
        else:
            compiler_status = (
                "READY_FOR_THEORY_GATES"
                if "READY_FOR_THEORY_GATES" in domain_admission.values()
                else "SOURCE_BLOCKED"
            )
        fingerprint = card["equivalence_fingerprint"]
        exact_family = content_sha256(fingerprint)
        compiled.append(
            {
                "concept_id": concept_id,
                "order_index": order_index,
                "entry_kind": "ATOMIC" if compound is None else "COMPOUND",
                "driver_ids": driver_ids,
                "architecture_id": architecture_id,
                "compiler_status": compiler_status,
                "domain_admission": domain_admission,
                "source_status_snapshot": source_snapshot,
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
                "equivalence_family_id": f"EQ-{exact_family[:24]}",
                "card_sha256": content_sha256(card),
                "card": card,
            }
        )
    return compiled


def cards_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def stream_root(rows: Sequence[Mapping[str, Any]]) -> str:
    return content_sha256([_sha256_bytes(_canonical(row)) for row in rows])


def receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_content_sha256", None)
    return content_sha256(payload)


def build_packet(root: Path | None = None) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    repo = _repo_root() if root is None else root.resolve()
    config = load_config(repo)
    (
        ledger,
        payloads,
        raw_config,
        module_bytes,
        test_bytes,
        failed_cards_bytes,
    ) = _bound_inputs(repo, config)
    _prior_art, _source = _validate_bound_metadata(config, payloads)
    prior_card_hashes = _failed_card_hashes(failed_cards_bytes)
    artifact_hashes = {
        "code_sha256": _sha256_bytes(module_bytes),
        "data_sha256": _sha256_bytes(payloads["SOURCE-AVAILABILITY"]),
        "environment_sha256": content_sha256(config["compiler_contract"]["environment_descriptor"]),
        "configuration_sha256": _sha256_bytes(raw_config),
    }
    rows = compile_rows(config, artifact_hashes, prior_card_hashes)
    payload = cards_bytes(rows)
    status_counts = dict(sorted(Counter(row["compiler_status"] for row in rows).items()))
    domain_counts = {
        domain: dict(sorted(Counter(row["domain_admission"][domain] for row in rows).items()))
        for domain in ("SPARC", "XCOP")
    }
    parameter_cell_count = sum(int(row["parameter_cell_count"]) for row in rows)
    parameter_cell_status_counts = dict(
        sorted(Counter(cell["status"] for row in rows for cell in row["cell_results"]).items())
    )
    parameter_cell_compiler_status_counts = dict(
        sorted(
            Counter(row["compiler_status"] for row in rows for _cell in row["cell_results"]).items()
        )
    )
    parameter_cell_domain_counts = {
        domain: dict(
            sorted(
                Counter(
                    row["domain_admission"][domain] for row in rows for _cell in row["cell_results"]
                ).items()
            )
        )
        for domain in ("SPARC", "XCOP")
    }
    if parameter_cell_count != int(
        config["probe_contract"]["expected_concept_parameter_cell_count"]
    ):
        raise TwellCompilerError("full Cartesian concept-parameter cell count changed")
    exact_groups: defaultdict[str, list[str]] = defaultdict(list)
    probe_groups: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        exact_groups[row["equivalence_family_id"]].append(row["concept_id"])
        probe_groups[row["probe_digest_sha256"]].append(row["concept_id"])
    probe_degeneracies = [group for group in probe_groups.values() if len(group) > 1]
    bound_hashes = {
        row["binding_id"]: row["sha256"] for row in config["governance_bindings"]["hard_bound"]
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "packet_id": PACKET_ID,
        "semantic_version": "1.1.0",
        "decision": DECISION,
        "enumeration": {
            "atomic_count": 380,
            "compound_count": 20,
            "total_count": len(rows),
            "ordered_concept_ids_sha256": content_sha256([row["concept_id"] for row in rows]),
            "first_id": rows[0]["concept_id"],
            "last_id": rows[-1]["concept_id"],
        },
        "compiler_status_counts": status_counts,
        "domain_admission_counts": domain_counts,
        "parameter_cell_summary": {
            "total_count": parameter_cell_count,
            "base_cartesian_count": config["probe_contract"]["base_cartesian_parameter_cell_count"],
            "compound_override_evidence_count": config["probe_contract"][
                "compound_override_evidence_cell_count"
            ],
            "probe_status_counts": parameter_cell_status_counts,
            "compiler_status_counts": parameter_cell_compiler_status_counts,
            "domain_admission_counts": parameter_cell_domain_counts,
        },
        "probe_summary": {
            "passed_card_count": sum(
                row["probe_status"] == "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES" for row in rows
            ),
            "failed_card_count": sum(
                row["probe_status"] != "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES" for row in rows
            ),
            "fixture_count_per_card": config["probe_contract"]["fixture_count_per_card"],
            "grid_points_per_fixture": config["probe_contract"]["fixture_grid_points"],
            "null_parameter_limit_passed_cell_count": parameter_cell_status_counts.get(
                "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES", 0
            ),
            "finite_probe_passed_cell_count": parameter_cell_status_counts.get(
                "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES", 0
            ),
            "deterministic_replay_passed_cell_count": parameter_cell_status_counts.get(
                "PASS_TARGET_FREE_EXACT_OPERATOR_PROBES", 0
            ),
            "maximum_computed_operator_residual": max(
                float(row["maximum_computed_operator_residual"]) for row in rows
            ),
            "maximum_computed_boundary_or_initial_residual": max(
                float(row["maximum_computed_boundary_or_initial_residual"]) for row in rows
            ),
            "maximum_computed_analytic_lambda_zero_residual": max(
                float(row["maximum_computed_analytic_lambda_zero_residual"]) for row in rows
            ),
            "checks_are_computed_evidence_not_copied_labels": True,
        },
        "equivalence_audit": {
            "exact_equivalence_family_count": len(exact_groups),
            "exact_collapsed_entry_count": sum(len(group) - 1 for group in exact_groups.values()),
            "probe_degeneracy_family_count": len(probe_degeneracies),
            "probe_degeneracy_entry_count": sum(len(group) for group in probe_degeneracies),
            "probe_degeneracy_groups": probe_degeneracies,
            "matching_probes_are_not_automatically_known_rewrites": True,
        },
        "closures": {
            "photon_default": "L0_NO_LIGHT_CLAIM",
            "capture_default": "C0_ISOLATED_CONSERVATIVE",
            "card_count_with_L0": sum(
                row["card"]["closures"]["photon"] == "L0_NO_LIGHT_CLAIM" for row in rows
            ),
            "card_count_with_C0": sum(
                row["card"]["closures"]["capture"] == "C0_ISOLATED_CONSERVATIVE" for row in rows
            ),
        },
        "stream": {
            "path": CARDS_PATH.as_posix(),
            "file_sha256": _sha256_bytes(payload),
            "ordered_line_root_sha256": stream_root(rows),
            "line_count": len(rows),
            "format": "canonical-jsonl",
        },
        "semantic_seals": dict(config["section_seals"]),
        "artifact_hashes": {
            "config_file_sha256": _sha256_bytes(raw_config),
            "config_canonical_sha256": content_sha256(config),
            "module_sha256": _sha256_bytes(module_bytes),
            "test_sha256": _sha256_bytes(test_bytes),
            "hard_bound_inputs": bound_hashes,
        },
        "deferred_bindings": list(config["governance_bindings"]["deferred_informational"]),
        "mechanism_schema_status": {
            "status": "DEFERRED_NON_AUTHORIZING_PENDING_FINAL_SCHEMA",
            "read_or_hashed": False,
            "internal_card_contract_used": True,
            "campaign_authority": False,
        },
        "retained_failed_packet": dict(config["output_contract"]["failed_packet_retained"]),
        "access_audit": {
            **dict(config["access_contract"]["zero_access"]),
            "runs_gravity_result_receipts_opened": 0,
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


def _rollback_owned_hardlink(temp_path: Path, destination: Path) -> None:
    """Remove destination only when it is provably the hard link created from temp."""

    try:
        owned = os.path.samestat(temp_path.stat(), destination.stat())
    except FileNotFoundError:
        return
    except OSError as error:
        raise TwellCompilerError(
            "could not prove ownership while rolling back partial TWELL publication"
        ) from error
    if not owned:
        raise TwellCompilerError(
            "refusing to remove a non-owned TWELL card artifact during rollback"
        )
    try:
        destination.unlink()
    except FileNotFoundError:
        return
    except OSError as error:
        raise TwellCompilerError("could not roll back owned TWELL card artifact") from error


def _atomic_packet_no_clobber(
    cards_path: Path, cards_payload: bytes, receipt_path: Path, receipt_payload: bytes
) -> None:
    """Atomically link both packet files, refusing to replace either destination."""

    if cards_path.exists() or receipt_path.exists():
        raise TwellCompilerError("refusing to overwrite an existing TWELL packet artifact")
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
        raise TwellCompilerError(
            "refusing to overwrite an existing TWELL packet artifact"
        ) from error
    except OSError as error:
        if cards_linked:
            _rollback_owned_hardlink(cards_temp, cards_path)
        raise TwellCompilerError("atomic TWELL packet creation failed") from error
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
        actual_receipt = _json_object(receipt_target.read_bytes(), str(receipt_target))
    except OSError as error:
        raise TwellCompilerError("could not read stored TWELL compiler packet") from error
    if actual_cards != expected_cards:
        raise TwellCompilerError("stored card stream differs from deterministic rebuild")
    if actual_receipt != expected_receipt:
        raise TwellCompilerError("stored receipt differs from deterministic rebuild")
    if actual_receipt["receipt_content_sha256"] != receipt_content_sha256(actual_receipt):
        raise TwellCompilerError("stored receipt self-hash failed")
    return actual_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--cards", type=Path, default=CARDS_PATH)
    build.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    check = subparsers.add_parser("check")
    check.add_argument("--cards", type=Path, default=CARDS_PATH)
    check.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    except TwellCompilerError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "total_count": receipt["enumeration"]["total_count"],
                "compiler_status_counts": receipt["compiler_status_counts"],
                "receipt_content_sha256": receipt["receipt_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
