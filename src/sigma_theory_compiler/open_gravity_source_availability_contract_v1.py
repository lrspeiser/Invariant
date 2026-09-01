"""Metadata-only source availability contract for the open-gravity campaign.

This module never reads a scientific response payload.  It expands a normalized
source contract into the exact mechanism x object x observable key space and
seals only counts and a canonical row hash.  It has no scoring, fitting,
candidate-selection, or campaign-freezing authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_source_availability_contract_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_source_availability_contract_v1.py")
TEST_PATH = Path("tests/test_open_gravity_source_availability_contract_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-source-availability-contract-v1.json")

SCHEMA = "invariant-open-gravity-source-availability-contract-1.0"
RECEIPT_SCHEMA = "invariant-open-gravity-source-availability-receipt-1.0"
EXPECTED_TWELL_IDS_SHA256 = "7388f8982c5014ef6c365d00aa780ba2ecb8b8b3f6786658fb3db36b64c29c5f"
EXPECTED_CONFIG_CONTENT_SHA256 = "e5e7d850b6db3cb8f6848ca9fbec56ce780359eae465adf967aa157d31ab9cb0"


class SourceAvailabilityError(RuntimeError):
    """Raised when the metadata-only source contract is inconsistent."""


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


def load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAvailabilityError(f"cannot load source contract: {path}") from exc
    validate_config(payload)
    return payload


def twell_concept_ids() -> list[str]:
    atomic = [
        f"TW2-A{architecture:02d}-D{driver:02d}"
        for architecture in range(1, 20)
        for driver in range(1, 21)
    ]
    return atomic + [f"X{compound:02d}" for compound in range(1, 21)]


def mechanism_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = config["mechanism_registry"]
    drivers = registry["twell"]["drivers"]
    architectures = registry["twell"]["architectures"]
    catalog: list[dict[str, Any]] = []
    for architecture_index, architecture in enumerate(architectures, start=1):
        for driver_index, driver in enumerate(drivers, start=1):
            catalog.append(
                {
                    "mechanism_id": f"TW2-A{architecture_index:02d}-D{driver_index:02d}",
                    "mechanism_family": "TWELL_ATOMIC",
                    "discovery_lane": "CORE",
                    "drivers": [driver],
                    "architecture": architecture,
                }
            )
    for compound in registry["twell"]["compounds"]:
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAvailabilityError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256,
        "source-availability config semantics changed",
    )
    _require(config.get("schema_version") == SCHEMA, "source schema changed")
    _require(config.get("append_only") is True, "append-only contract changed")
    authority = config["authority_boundary"]
    _require(authority["campaign_manifest_frozen"] is False, "campaign was frozen here")
    for forbidden in (
        "may_choose_or_repair_formulas",
        "may_score_responses",
        "may_set_promotion_thresholds",
        "may_adjudicate_candidates",
    ):
        _require(authority[forbidden] is False, f"forbidden authority enabled: {forbidden}")
    _require(len(config["incident_ledger"]) == 2, "exposure incident ledger changed")

    registry = config["mechanism_registry"]
    twell = registry["twell"]
    _require(len(twell["drivers"]) == 20, "TWELL driver count changed")
    _require(len(twell["architectures"]) == 19, "TWELL architecture count changed")
    _require(len(twell["compounds"]) == 20, "TWELL compound count changed")
    _require(
        content_sha256(twell_concept_ids()) == EXPECTED_TWELL_IDS_SHA256, "TWELL ID root changed"
    )
    _require(
        twell["ordered_concept_ids_sha256"] == EXPECTED_TWELL_IDS_SHA256, "TWELL binding changed"
    )
    _require(len(registry["GP01_variants"]) == 7, "GP01 branch count changed")
    _require(len(registry["ontology_nodes"]) == 13, "ontology count changed")
    _require(len(registry["discovery_lanes"]) == 5, "discovery lane count changed")
    _require(len(mechanism_catalog(config)) == 420, "mechanism catalog is not 420")

    sparc = config["objects"]["SPARC"]
    xcop = config["objects"]["XCOP"]
    _require(len(sparc) == len(set(sparc)) == 139, "SPARC object ledger changed")
    _require(len(xcop) == len(set(xcop)) == 8, "X-COP object ledger changed")
    available = set(config["objects"]["XCOP_stellar_profile_available"])
    missing = set(config["objects"]["XCOP_stellar_profile_missing"])
    _require(available.isdisjoint(missing), "stellar source roles overlap")
    _require(available | missing == set(xcop), "stellar source roles do not cover X-COP")

    driver_ids = set(twell["drivers"])
    for domain in ("SPARC", "XCOP"):
        _require(
            set(config["driver_source_availability"][domain]) == driver_ids,
            f"{domain} driver availability is incomplete",
        )
    _require(
        set(config["GP01_source_availability"]) == set(registry["GP01_variants"]),
        "GP01 source matrix changed",
    )
    _require(
        set(config["ontology_source_availability"]) == set(registry["ontology_nodes"]),
        "ontology source matrix changed",
    )
    statuses = set(config["source_status_vocabulary"])
    for domain_rows in config["driver_source_availability"].values():
        _require(set(domain_rows.values()) <= statuses, "unknown driver source status")
    for branch_rows in config["GP01_source_availability"].values():
        _require(set(branch_rows.values()) <= statuses, "unknown GP01 source status")
    for node_rows in config["ontology_source_availability"].values():
        _require(set(node_rows.values()) <= statuses, "unknown ontology source status")

    pilot = config["partition_design"]["SPARC_pilot"]
    _require(len(pilot) == len(set(pilot)) == 28, "SPARC pilot changed")
    _require(set(pilot) < set(sparc), "SPARC pilot is not a strict subset")
    _require(
        hashlib.sha256("\n".join(pilot).encode("utf-8")).hexdigest()
        == config["partition_design"]["SPARC_pilot_rank_order_sha256"],
        "SPARC pilot root changed",
    )
    xcop_pilot = config["partition_design"]["XCOP_pilot"]
    xcop_validation = config["partition_design"]["XCOP_validation"]
    _require(set(xcop_pilot).isdisjoint(xcop_validation), "X-COP partitions overlap")
    _require(set(xcop_pilot) | set(xcop_validation) == set(xcop), "X-COP split changed")
    _require(
        hashlib.sha256("\n".join(xcop_pilot).encode("utf-8")).hexdigest()
        == config["partition_design"]["XCOP_pilot_rank_order_sha256"],
        "X-COP pilot root changed",
    )
    _require(
        hashlib.sha256("\n".join(xcop_validation).encode("utf-8")).hexdigest()
        == config["partition_design"]["XCOP_validation_rank_order_sha256"],
        "X-COP validation root changed",
    )
    matrix = config["matrix_contract"]
    _require(matrix["expanded_tuple_count"] == 65100, "matrix tuple count changed")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


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


def _combined_driver_status(config: Mapping[str, Any], domain: str, drivers: Sequence[str]) -> str:
    if not drivers:
        return "OUT_OF_SCOPE_NOT_OPENED"
    statuses = [config["driver_source_availability"][domain][driver] for driver in drivers]
    return max(statuses, key=_STATUS_PRECEDENCE.__getitem__)


def _mechanism_source_status(
    config: Mapping[str, Any], mechanism: Mapping[str, Any], domain: str, object_id: str
) -> str:
    family = mechanism["mechanism_family"]
    mechanism_id = mechanism["mechanism_id"]
    if family == "TWELL_ATOMIC":
        return "UNKNOWN_SOURCE_BLOCKED"
    if family == "TWELL_COMPOUND":
        if mechanism_id != "X01":
            return "UNKNOWN_SOURCE_BLOCKED"
        if domain == "SPARC":
            return "SOURCE_MISSING"
        if object_id in config["objects"]["XCOP_stellar_profile_missing"]:
            return "SOURCE_AVAILABLE_WITH_SHARED_GLOBAL_STELLAR_NUISANCE"
        return "SOURCE_AVAILABLE_SPHERICAL_RADIAL_ONLY"
    if family == "GP01":
        status = config["GP01_source_availability"][mechanism_id][domain]
    else:
        status = config["ontology_source_availability"][mechanism_id][domain]
    if (
        domain == "XCOP"
        and status == "SOURCE_AVAILABLE_WITH_SHARED_GLOBAL_STELLAR_NUISANCE"
        and object_id not in config["objects"]["XCOP_stellar_profile_missing"]
    ):
        return "SOURCE_AVAILABLE_SPHERICAL_RADIAL_ONLY"
    return status


def iter_matrix_rows(config: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    observables = config["observable_contracts"]
    stellar_available = set(config["objects"]["XCOP_stellar_profile_available"])
    for mechanism in mechanism_catalog(config):
        for object_id in config["objects"]["SPARC"]:
            observable = observables["SPARC"][0]
            yield {
                "mechanism_id": mechanism["mechanism_id"],
                "mechanism_family": mechanism["mechanism_family"],
                "discovery_lane": mechanism["discovery_lane"],
                "object_id": object_id,
                "domain": "SPARC",
                "observable_id": observable["observable_id"],
                "driver_source_status": _combined_driver_status(
                    config, "SPARC", mechanism["drivers"]
                ),
                "mechanism_source_status": _mechanism_source_status(
                    config, mechanism, "SPARC", object_id
                ),
                "observable_source_status": observable["status"],
                "stellar_source_status": "OUT_OF_SCOPE_NOT_OPENED",
                "shared_provenance": False,
                "source_only_no_scoring_authority": True,
            }
        for object_id in config["objects"]["XCOP"]:
            for observable in observables["XCOP"]:
                yield {
                    "mechanism_id": mechanism["mechanism_id"],
                    "mechanism_family": mechanism["mechanism_family"],
                    "discovery_lane": mechanism["discovery_lane"],
                    "object_id": object_id,
                    "domain": "XCOP",
                    "observable_id": observable["observable_id"],
                    "driver_source_status": _combined_driver_status(
                        config, "XCOP", mechanism["drivers"]
                    ),
                    "mechanism_source_status": _mechanism_source_status(
                        config, mechanism, "XCOP", object_id
                    ),
                    "observable_source_status": observable["status"],
                    "stellar_source_status": (
                        "SOURCE_AVAILABLE" if object_id in stellar_available else "SOURCE_MISSING"
                    ),
                    "shared_provenance": True,
                    "source_only_no_scoring_authority": True,
                }


def matrix_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    mechanism_statuses: Counter[str] = Counter()
    driver_statuses: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    for row in iter_matrix_rows(config):
        digest.update(_canonical(row))
        digest.update(b"\n")
        count += 1
        mechanism_statuses[row["mechanism_source_status"]] += 1
        driver_statuses[row["driver_source_status"]] += 1
        domain_counts[row["domain"]] += 1
    _require(count == config["matrix_contract"]["expanded_tuple_count"], "matrix count mismatch")
    return {
        "expanded_tuple_count": count,
        "canonical_row_stream_sha256": digest.hexdigest(),
        "domain_tuple_counts": dict(sorted(domain_counts.items())),
        "mechanism_source_status_counts": dict(sorted(mechanism_statuses.items())),
        "driver_source_status_counts": dict(sorted(driver_statuses.items())),
        "rows_materialized_in_receipt": 0,
        "normalization": config["matrix_contract"]["normalization"],
    }


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    catalog = mechanism_catalog(config)
    sparc = config["objects"]["SPARC"]
    xcop = config["objects"]["XCOP"]
    pilot = config["partition_design"]["SPARC_pilot"]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_id": config["contract_id"],
        "status": config["status"],
        "authority_boundary": config["authority_boundary"],
        "dependency_audit_gates": config["dependency_audit_gates"],
        "zero_access": {
            "scientific_response_payloads_read_by_generator": 0,
            "confirmation_rows_read": 0,
            "independent_rows_read": 0,
            "group_rows_read": 0,
            "lensing_rows_read": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "scores_computed": 0,
            "candidates_selected_or_repaired": 0,
        },
        "source_auditor_incidents": config["incident_ledger"],
        "bindings": {
            "config": {"path": CONFIG_PATH.as_posix(), "sha256": file_sha256(root / CONFIG_PATH)},
            "module": {"path": MODULE_PATH.as_posix(), "sha256": file_sha256(root / MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": file_sha256(root / TEST_PATH)},
            "goal_documents": config["goal_bindings"],
        },
        "catalog": {
            "mechanisms": len(catalog),
            "TWELL": 400,
            "GP01": 7,
            "ontology_nodes": 13,
            "discovery_lanes": config["mechanism_registry"]["discovery_lanes"],
            "lane_entry_counts": dict(
                sorted(Counter(row["discovery_lane"] for row in catalog).items())
            ),
            "empty_frozen_lanes": ["ADJACENT"],
            "twell_ordered_ids_sha256": content_sha256(twell_concept_ids()),
        },
        "objects": {
            "SPARC": len(sparc),
            "XCOP": len(xcop),
            "total": len(sparc) + len(xcop),
            "SPARC_ordered_membership_sha256": content_sha256(sparc),
            "XCOP_ordered_membership_sha256": content_sha256(xcop),
            "ledger_seals": config["object_ledger_seals"],
            "XCOP_stellar_profile_available": config["objects"]["XCOP_stellar_profile_available"],
            "XCOP_stellar_profile_missing": config["objects"]["XCOP_stellar_profile_missing"],
        },
        "observables": config["observable_contracts"],
        "source_availability_rules": {
            "drivers": config["driver_source_availability"],
            "GP01": config["GP01_source_availability"],
            "ontology": config["ontology_source_availability"],
            "architecture_notes": config["architecture_source_notes"],
            "GP01_notes": config["GP01_source_notes"],
        },
        "matrix": matrix_summary(config),
        "comparators": config["comparator_inventory"],
        "comparator_code_inventory": config["comparator_code_inventory"],
        "source_bindings": config["source_bindings"],
        "partition_design": {
            "status": config["partition_design"]["status"],
            "SPARC_pilot_count": len(pilot),
            "SPARC_validation_count": len(sparc) - len(pilot),
            "SPARC_pilot_rank_order_sha256": config["partition_design"][
                "SPARC_pilot_rank_order_sha256"
            ],
            "SPARC_validation_rank_order_sha256": config["partition_design"][
                "SPARC_validation_rank_order_sha256"
            ],
            "SPARC_all_rank_order_sha256": config["partition_design"][
                "SPARC_all_rank_order_sha256"
            ],
            "XCOP_pilot": config["partition_design"]["XCOP_pilot"],
            "XCOP_validation": config["partition_design"]["XCOP_validation"],
            "XCOP_pilot_rank_order_sha256": config["partition_design"][
                "XCOP_pilot_rank_order_sha256"
            ],
            "XCOP_validation_rank_order_sha256": config["partition_design"][
                "XCOP_validation_rank_order_sha256"
            ],
            "XCOP_all_rank_order_sha256": config["partition_design"]["XCOP_all_rank_order_sha256"],
        },
        "legacy_multiplicity_lower_bound": config["legacy_multiplicity_lower_bound"],
        "campaign_manifest": {
            "status": "UNFROZEN",
            "reason": "registry_and_GP01_audits_and_independent_candidate_fixing_are_not_complete",
            "response_execution_authorized_by_this_receipt": False,
        },
        "out_of_scope": config["out_of_scope"],
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
        raise SourceAvailabilityError(f"refusing to overwrite append-only receipt: {path}")
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return "CREATED"


def write_receipt(root: Path) -> str:
    receipt = build_receipt(root)
    payload = (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    return _atomic_no_clobber(root / OUTPUT_PATH, payload)


def validate_receipt(root: Path) -> None:
    path = root / OUTPUT_PATH
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAvailabilityError(f"cannot load source receipt: {path}") from exc
    observed = stored.pop("content_sha256", None)
    _require(observed == content_sha256(stored), "receipt content hash changed")
    expected = build_receipt(root)
    _require({**stored, "content_sha256": observed} == expected, "receipt is not reproducible")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(write_receipt(args.root.resolve()))
    else:
        validate_receipt(args.root.resolve())
        print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
