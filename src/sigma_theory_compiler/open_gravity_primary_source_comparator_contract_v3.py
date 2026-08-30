"""Append-only canonical-path registry reseal of the open-gravity prior-art contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_primary_source_comparator_contract_v3.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_primary_source_comparator_contract_v3.py"
)
TEST_PATH = Path("tests/test_open_gravity_primary_source_comparator_contract_v3.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-primary-source-comparator-contract-v3/receipt.json")
EXPECTED_OUTPUT_PATH = (
    "runs/gravity/open-gravity-primary-source-comparator-contract-v3/receipt.json"
)
CONTRACT_ID = "OPEN-GRAVITY-PRIMARY-SOURCE-COMPARATOR-CONTRACT-v3"
DECISION = (
    "PASS_FINAL_REGISTRY_REBOUND_CANONICAL_PATH_PRIOR_ART_METADATA_ONLY_NO_CAMPAIGN_AUTHORITY"
)
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "4dd3c9202804eefae22e0e7ff06a1d03d2a1bc5e628f139655715035008184ea"
)
EXPECTED_UNSEALED_ROOT_SHA256 = "611868e53cad62e2dd4ac037586b4376c2123a1991c268083f9d77b5868be93c"
EXPECTED_SECTION_SEALS = {
    "identity": "5cd1e1e6651f2a957a3f71e8ee4efe6c867bcb354a0b50c3712f4aefa723277b",
    "hard_bindings": "fdc593ac25be04f30302973c1fb67c33962146f7084f3d6320e607ec7daa7c5b",
    "commit_provenance": "94fa11ffe883451ff4ea8bb37a4f218317ca8a79eb03bf4d83a46477e35f51da",
    "payload_contract": "ff012ab794ec35acc064eb897bd0363688b589d06bd4e53a40629953bfec6649",
    "access_contract": "14232750c1c5742b87aae75a4e99de17094cc09d8d9a569dce7eddeaea579941",
    "claim_boundary": "49e88a991cdfd3c49b469c9bb8f79c69d2019b5c2053490d42d47846f1ae3a89",
    "output_contract": "2bd133d30c59ce71c95947e352a2d2ba4e2291fcbce241340b5b48507c222432",
}


class PriorArtResealError(RuntimeError):
    """Raised when the final prior-art reseal fails closed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def receipt_content_sha256(receipt: dict[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("content_sha256", None)
    return content_sha256(payload)


def validate_config(config: dict[str, Any]) -> None:
    expected_sections = set(EXPECTED_SECTION_SEALS)
    if set(config) != expected_sections | {"section_seals"}:
        raise PriorArtResealError("config top-level sections changed")
    seals = config["section_seals"]
    if set(seals) != expected_sections | {"unsealed_root_sha256"}:
        raise PriorArtResealError("config seal inventory changed")
    for section, expected in EXPECTED_SECTION_SEALS.items():
        if seals[section] != expected or content_sha256(config[section]) != expected:
            raise PriorArtResealError(f"sealed section changed: {section}")
    unsealed = {key: value for key, value in config.items() if key != "section_seals"}
    if (
        seals["unsealed_root_sha256"] != EXPECTED_UNSEALED_ROOT_SHA256
        or content_sha256(unsealed) != EXPECTED_UNSEALED_ROOT_SHA256
    ):
        raise PriorArtResealError("unsealed config root changed")
    if content_sha256(config) != EXPECTED_CONFIG_CANONICAL_SHA256:
        raise PriorArtResealError("canonical config hash changed")
    identity = config["identity"]
    if (
        identity.get("contract_id") != CONTRACT_ID
        or identity.get("semantic_version") != "3.0.0"
        or identity.get("append_only") is not True
        or identity.get("source_content_revision") is not False
    ):
        raise PriorArtResealError("identity contract changed")
    rows = config["hard_bindings"]
    identifiers = [row["binding_id"] for row in rows]
    if len(rows) != 13 or len(identifiers) != len(set(identifiers)):
        raise PriorArtResealError("hard-binding inventory changed")
    if config["claim_boundary"].get("campaign_execution_authority") is not False:
        raise PriorArtResealError("prior-art reseal cannot authorize a campaign")
    if any(config["access_contract"]["zero_access"].values()):
        raise PriorArtResealError("zero-access contract changed")
    output = config["output_contract"]
    if (
        output.get("receipt_path") != EXPECTED_OUTPUT_PATH
        or output.get("check_accepts_path_override") is not False
        or output.get("cli_output_override_allowed") is not False
    ):
        raise PriorArtResealError("canonical output-path contract changed")


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    try:
        config = json.loads((repo / CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PriorArtResealError("could not load prior-art reseal config") from error
    if not isinstance(config, dict):
        raise PriorArtResealError("prior-art reseal config must be an object")
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
            raise PriorArtResealError(f"non-allowlisted metadata read refused: {resolved}")
        try:
            payload = resolved.read_bytes()
        except OSError as error:
            raise PriorArtResealError(f"could not read metadata: {resolved}") from error
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
        raise PriorArtResealError(f"invalid JSON: {label}") from error
    if not isinstance(value, dict):
        raise PriorArtResealError(f"JSON object required: {label}")
    return value


def _verify_self_hash(
    value: dict[str, Any], field_name: str, label: str, *, trailing_newline: bool = False
) -> None:
    payload = dict(value)
    claimed = payload.pop(field_name, None)
    canonical = _canonical(payload) + (b"\n" if trailing_newline else b"")
    if claimed != _sha256_bytes(canonical):
        raise PriorArtResealError(f"{label} self-hash failed")


def _validate_payloads(
    config: dict[str, Any], payloads: dict[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prior = _json_object(payloads["PRIOR-ART-v1-PAYLOAD"], "prior-art v1 payload")
    registry_receipt = _json_object(payloads["REGISTRY-RECEIPT-FINAL"], "registry receipt")
    mechanism_schema = _json_object(payloads["MECHANISM-CARD-SCHEMA-FINAL"], "mechanism schema")
    gp01_receipt = _json_object(payloads["GP01-RECEIPT-FINAL"], "GP01 receipt")
    binding_hashes = {row["binding_id"]: row["sha256"] for row in config["hard_bindings"]}
    contract = config["payload_contract"]
    if len(prior.get("primary_sources", [])) != contract["primary_source_count"]:
        raise PriorArtResealError("primary-source count changed")
    comparator_ids = [row["comparator_id"] for row in prior["dynamical_comparators"]]
    ontology_ids = [row["ontology_id"] for row in prior["ontology_prior_art"]]
    analogy_ids = [row["analogy_id"] for row in prior["light_gravity_analogies"]]
    if comparator_ids != contract["required_comparator_ids"]:
        raise PriorArtResealError("comparator inventory changed")
    if ontology_ids != contract["required_ontology_ids"]:
        raise PriorArtResealError("QG ontology inventory changed")
    if analogy_ids != contract["required_analogy_ids"]:
        raise PriorArtResealError("light-gravity analogy inventory changed")
    if any(
        row.get("never_substitute_name_only") is not True for row in prior["dynamical_comparators"]
    ):
        raise PriorArtResealError("name-only comparator substitution guard changed")
    if registry_receipt.get("decision") != (
        "PASS_OPEN_GRAVITY_REGISTRY_FOUNDATION_ZERO_RESPONSE_ACCESS_NO_SCIENTIFIC_SCORE"
    ):
        raise PriorArtResealError("registry receipt decision changed")
    if (
        registry_receipt["bindings"]["config_file_sha256"]
        != binding_hashes["REGISTRY-CONFIG-FINAL"]
    ):
        raise PriorArtResealError("registry config/receipt binding mismatch")
    if (
        registry_receipt["bindings"]["schemas"]["mechanism_card"]["file_sha256"]
        != binding_hashes["MECHANISM-CARD-SCHEMA-FINAL"]
    ):
        raise PriorArtResealError("mechanism schema/registry binding mismatch")
    if mechanism_schema.get("$id") != "urn:invariant:open-gravity:mechanism-card:1.0":
        raise PriorArtResealError("mechanism schema identity changed")
    _verify_self_hash(registry_receipt, "content_sha256", "registry receipt")
    _verify_self_hash(gp01_receipt, "content_sha256", "GP01 receipt", trailing_newline=True)
    if gp01_receipt.get("decision") != (
        "GP01_FOUNDATION_PASS_SYNTHETIC_ONLY_ACTION_AND_CAUSAL_COMPLETION_QUARANTINED"
    ):
        raise PriorArtResealError("GP01 receipt decision changed")
    return prior, registry_receipt, gp01_receipt


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    config = load_config(repo)
    allowed = {
        repo / CONFIG_PATH: "reseal_config",
        repo / MODULE_PATH: "reseal_module",
        repo / TEST_PATH: "reseal_tests",
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
            raise PriorArtResealError(f"hard binding changed: {row['binding_id']}")
        payloads[row["binding_id"]] = payload
    prior, registry_receipt, gp01_receipt = _validate_payloads(config, payloads)
    binding_rows = {row["binding_id"]: row for row in config["hard_bindings"]}
    receipt: dict[str, Any] = {
        "schema_version": "invariant-open-gravity-primary-source-comparator-reseal-receipt-3.0",
        "contract_id": CONTRACT_ID,
        "semantic_version": "3.0.0",
        "decision": DECISION,
        "status": "MUTATION_FROZEN_FINAL_CANONICAL_PATH_REGISTRY_REBOUND_PRIOR_ART",
        "payload": {
            "path": binding_rows["PRIOR-ART-v1-PAYLOAD"]["path"],
            "file_sha256": binding_rows["PRIOR-ART-v1-PAYLOAD"]["sha256"],
            "semantic_content_sha256": content_sha256(prior),
            "source_content_mutated": False,
            "primary_source_count": len(prior["primary_sources"]),
            "dynamical_comparator_count": len(prior["dynamical_comparators"]),
            "ontology_prior_art_count": len(prior["ontology_prior_art"]),
            "light_gravity_analogy_count": len(prior["light_gravity_analogies"]),
        },
        "registry_rebind": {
            "commit": config["commit_provenance"]["registry_commit"],
            "config_sha256": registry_receipt["bindings"]["config_file_sha256"],
            "receipt_sha256": binding_rows["REGISTRY-RECEIPT-FINAL"]["sha256"],
            "mechanism_schema_sha256": binding_rows["MECHANISM-CARD-SCHEMA-FINAL"]["sha256"],
            "mechanism_schema_semantic_sha256": registry_receipt["bindings"]["schemas"][
                "mechanism_card"
            ]["semantic_content_sha256"],
        },
        "gp01_rebind": {
            "commit": config["commit_provenance"]["gp01_commit"],
            "config_sha256": binding_rows["GP01-CONFIG-FINAL"]["sha256"],
            "receipt_sha256": binding_rows["GP01-RECEIPT-FINAL"]["sha256"],
            "receipt_content_sha256": gp01_receipt["content_sha256"],
        },
        "predecessor": {
            "blocked_v2_config_sha256": binding_rows["BLOCKED-PRIOR-ART-v2-CONFIG"]["sha256"],
            "blocked_v2_module_sha256": binding_rows["BLOCKED-PRIOR-ART-v2-MODULE"]["sha256"],
            "blocked_v2_test_sha256": binding_rows["BLOCKED-PRIOR-ART-v2-TEST"]["sha256"],
            "blocked_v2_receipt_sha256": binding_rows["BLOCKED-PRIOR-ART-v2-RECEIPT"]["sha256"],
            "status": "BLOCKED_CANONICAL_PATH_AUDIT_COUNTEREVIDENCE_NO_AUTHORITY",
            "scientific_source_content_reused_unchanged": True,
        },
        "canonical_path_enforcement": {
            "receipt_path": EXPECTED_OUTPUT_PATH,
            "check_path_override_accepted": False,
            "cli_output_override_accepted": False,
            "path_verified_before_receipt_read": True,
        },
        "semantic_seals": dict(config["section_seals"]),
        "artifact_hashes": {
            "config_file_sha256": _sha256_bytes(raw_config),
            "config_canonical_sha256": content_sha256(config),
            "module_sha256": _sha256_bytes(module_bytes),
            "test_sha256": _sha256_bytes(test_bytes),
            "hard_bindings": {row["binding_id"]: row["sha256"] for row in config["hard_bindings"]},
        },
        "access_audit": {
            **config["access_contract"]["zero_access"],
            "allowlisted_metadata_files_opened": ledger.rows(),
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    receipt["content_sha256"] = receipt_content_sha256(receipt)
    return receipt


def _write_no_clobber(path: Path, payload: bytes) -> None:
    if path.exists():
        raise PriorArtResealError("refusing to overwrite prior-art reseal receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PriorArtResealError("refusing to overwrite prior-art reseal receipt") from error
    except OSError as error:
        raise PriorArtResealError("atomic prior-art reseal publication failed") from error
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(receipt: dict[str, Any], path: Path) -> None:
    _write_no_clobber(path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())


def _canonical_receipt_path(repo: Path) -> Path:
    if OUTPUT_PATH.as_posix() != EXPECTED_OUTPUT_PATH:
        raise PriorArtResealError("canonical prior-art receipt path constant changed")
    target = repo / Path(EXPECTED_OUTPUT_PATH)
    if target.absolute() != (repo / OUTPUT_PATH).absolute():
        raise PriorArtResealError("canonical prior-art receipt path mismatch")
    return target


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    target = _canonical_receipt_path(repo)
    expected = build_receipt(repo)
    try:
        actual = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PriorArtResealError("could not read stored prior-art reseal receipt") from error
    if actual != expected or actual.get("content_sha256") != receipt_content_sha256(actual):
        raise PriorArtResealError("stored prior-art reseal receipt differs from rebuild")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    args = parser.parse_args(argv)
    repo = _repo_root()
    try:
        if args.command == "build":
            receipt = build_receipt(repo)
            write_receipt(receipt, _canonical_receipt_path(repo))
        else:
            receipt = check_receipt(repo)
    except PriorArtResealError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {"decision": receipt["decision"], "content_sha256": receipt["content_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
