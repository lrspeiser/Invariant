"""Exact recovery ledger for gravity concepts sidelined without an empirical loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_audit_recovery_program_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_audit_recovery_program_v1.py")
TEST_PATH = Path("tests/test_open_gravity_audit_recovery_program_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-audit-recovery-program-v1/receipt.json")
_CANONICAL_OUTPUT_PATH = Path("runs/gravity/open-gravity-audit-recovery-program-v1/receipt.json")
_SCHEMA = "invariant-open-gravity-audit-recovery-program-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-audit-recovery-program-receipt-1.0"
_CONFIG_CONTENT_SHA256 = "0ef076dfd177a9f59002a70d30b1b69f18f1b13f8c95e619a23526c3e12ac31b"
_EXPECTED_CLASS_COUNTS = {
    "RC01_TWELL_STATIC_MISSING_DRIVERS": 180,
    "RC02_TWELL_DYNAMIC_HISTORY": 80,
    "RC03_TWELL_COMPOUNDS": 14,
    "RC04_GP01_TRANSPORT": 3,
    "RC05_GP01_AQUAL_CONTROL": 1,
    "RC06_GP01_ACTION_REPAIR": 1,
}
_DYNAMIC_ARCHITECTURES = {15, 16, 17, 18}
_GP01_CLASSES = {
    "GP01-T1": "RC04_GP01_TRANSPORT",
    "GP01-T2": "RC04_GP01_TRANSPORT",
    "GP01-TELEGRAPH": "RC04_GP01_TRANSPORT",
    "GP01-AQUAL": "RC05_GP01_AQUAL_CONTROL",
    "GP01-ACTION-PLACEHOLDER": "RC06_GP01_ACTION_REPAIR",
}


class AuditRecoveryError(RuntimeError):
    """Raised when recovery coverage or a frozen binding fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditRecoveryError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditRecoveryError(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    required = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "dual_grade_policy",
        "latest_campaign_recovery_classes",
        "historical_recovery_workstreams",
        "publication_lead_policy",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(set(config) == required, "config keys changed")
    _require(config["schema"] == _SCHEMA, "config schema changed")
    _require(config["package_id"] == "open-gravity-audit-recovery-program-v1", "ID changed")
    _require(
        config["status"] == "ACTIVE_RECOVERY_NO_THEORY_ELIMINATED_BY_NONEMPIRICAL_GATE",
        "status changed",
    )
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output path changed")
    classes = config["latest_campaign_recovery_classes"]
    _require(type(classes) is list and len(classes) == 6, "recovery class inventory changed")
    observed = {row["id"]: row["expected_concepts"] for row in classes}
    _require(observed == _EXPECTED_CLASS_COUNTS, "recovery class counts changed")
    for row in classes:
        _require(row["empirical_grade"].startswith("UNTESTED"), "blocked class called tested")
        _require(bool(row["empirical_next"]), "empirical next step missing")
        _require(bool(row["theory_next"]), "theory next step missing")
        _require(bool(row["publication_hook"]), "publication hook missing")
    historical = config["historical_recovery_workstreams"]
    _require(type(historical) is list and len(historical) == 12, "historical inventory changed")
    _require(len({row["id"] for row in historical}) == len(historical), "duplicate workstream")
    _require(all(row["status"].startswith("ACTIVE") for row in historical), "family eliminated")
    access = config["access_contract"]
    _require(set(access.values()) == {0}, "nonzero access authorized")
    policy = config["dual_grade_policy"]
    _require(
        "DATA_REJECTED_EXACT_IMPLEMENTATION" in policy["empirical_grades"], "data loss grade absent"
    )
    _require("THEORY_OBSTRUCTION_EXACT_BRANCH" in policy["theory_grades"], "theory grade absent")
    _require(
        policy["rule"].startswith("A theory grade never overwrites"), "dual-grade rule changed"
    )


def load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH, "recovery config")
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for binding in config["bindings"]:
        path = Path(binding["path"])
        _require(path.is_file(), f"missing binding: {binding['role']}")
        digest = file_sha256(path)
        _require(digest == binding["sha256"], f"binding changed: {binding['role']}")
        observed[binding["role"]] = digest
    _require(len(observed) == 7, "binding inventory changed")
    return observed


def classify_concept(concept_id: str) -> str:
    if concept_id in _GP01_CLASSES:
        return _GP01_CLASSES[concept_id]
    if concept_id.startswith("X"):
        return "RC03_TWELL_COMPOUNDS"
    parts = concept_id.split("-")
    _require(len(parts) == 3 and parts[0] == "TW2", f"unclassified concept: {concept_id}")
    _require(parts[1].startswith("A") and parts[2].startswith("D"), "bad TWELL concept")
    architecture = int(parts[1][1:])
    driver = int(parts[2][1:])
    if architecture in _DYNAMIC_ARCHITECTURES:
        _require(1 <= driver <= 20, "dynamic driver outside registry")
        return "RC02_TWELL_DYNAMIC_HISTORY"
    _require(architecture in {*range(1, 15), 19}, "unexpected static architecture")
    _require(driver in {*range(8, 13), *range(14, 21)}, "unexpected static blocked driver")
    return "RC01_TWELL_STATIC_MISSING_DRIVERS"


def _coverage(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = next(row for row in config["bindings"] if row["role"] == "BLOCKED_IDEA_LEDGER")
    ledger = _read_json(Path(binding["path"]), "blocked idea ledger")
    _require(type(ledger) is list and len(ledger) == 279, "blocked ledger count changed")
    ids = [row["concept_id"] for row in ledger]
    _require(len(set(ids)) == 279, "blocked ledger concept duplication")
    assignments = [
        {
            "concept_id": row["concept_id"],
            "prior_candidate_status": row["candidate_status"],
            "recovery_class": classify_concept(row["concept_id"]),
            "empirical_result": "NO_EMPIRICAL_LOSS_IN_BOUND_CAMPAIGN",
        }
        for row in ledger
    ]
    counts = Counter(row["recovery_class"] for row in assignments)
    _require(dict(sorted(counts.items())) == _EXPECTED_CLASS_COUNTS, "coverage counts changed")
    status_counts = Counter(row["candidate_status"] for row in ledger)
    _require(
        dict(sorted(status_counts.items()))
        == {
            "KNOWN_REWRITE_NONINDEPENDENT": 1,
            "QUARANTINED_REVISION_REQUIRED": 1,
            "REGISTERED_THEORY_ONLY": 274,
            "SOURCE_BLOCKED": 3,
        },
        "prior statuses changed",
    )
    return {
        "concepts": len(assignments),
        "class_counts": dict(sorted(counts.items())),
        "prior_status_counts": dict(sorted(status_counts.items())),
        "assignment_root_sha256": content_sha256(assignments),
        "assignments": assignments,
    }


def _publication_leads(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    binding = next(row for row in config["bindings"] if row["role"] == "GP01_EMPIRICAL_SCREEN")
    screen = _read_json(Path(binding["path"]), "GP01 screen")["screen"]
    entropy = screen["dynamical_proxy_correlations"]["LOG_K0_KEV_CM2"]
    elliptic = screen["elliptic"]
    _require(elliptic["beats_equilibrium_on_objects"] == 0, "counterexample erased")
    _require(elliptic["object_count"] == 8, "screen object count changed")
    _require(
        entropy["loss_difference"]["exact_two_sided_p"] == 0.00873015873015873,
        "entropy result changed",
    )
    return [
        {
            "lead_id": "PL01_CLUSTER_DYNAMICAL_STATE_RESIDUAL",
            "state": "DEVELOPMENT_ASSOCIATION",
            "exact_claim": "On eight development clusters, the failure size of the frozen GP01 elliptic branch is associated with central entropy, while the branch loses to equilibrium on every object.",
            "closest_published_neighbor": "MOND cluster residual-mass and hydrostatic nonequilibrium literature",
            "unique_discriminator": "Pre-registered leave-one-cluster-out prediction of residual change from a source-history or relaxation proxy at fixed local baryonic field.",
            "counterexample": "GP01 elliptic beats equilibrium on 0 of 8 objects.",
            "empirical_grade": "DATA_REJECTED_EXACT_IMPLEMENTATION_WITH_RETROSPECTIVE_ASSOCIATION",
            "theory_grade": "PHENOMENOLOGY_ONLY",
            "next_falsifier": "The association disappears in a larger frozen cluster sample or a shared history parameter fails leave-one-out prediction.",
            "evidence": {
                "objects": 8,
                "entropy_loss_difference_spearman_rho": entropy["loss_difference"]["rho"],
                "entropy_loss_difference_exact_p": entropy["loss_difference"]["exact_two_sided_p"],
                "elliptic_to_equilibrium_robust_loss_ratio": elliptic["robust_loss_ratio"],
            },
        },
        {
            "lead_id": "PL02_TEMPORAL_TRANSFER_FINGERPRINTS",
            "state": "THEORY_SIGNATURE_ONLY",
            "exact_claim": "Retardation, exponential memory, damped resonance, and stochastic relaxation must have distinguishable causal transfer signatures before real-response scoring.",
            "closest_published_neighbor": "Causal nonlocal gravity, fading-memory phenomenology, and linear response theory",
            "unique_discriminator": "Frequency-dependent phase, attenuation, hysteresis, resonant amplification, and seeded variance on identical source histories.",
            "counterexample": "Static snapshots cannot identify a temporal kernel.",
            "empirical_grade": "UNTESTED_SOURCE_BLOCKED",
            "theory_grade": "PHENOMENOLOGY_ONLY",
            "next_falsifier": "Two registered temporal architectures remain observationally and synthetically indistinguishable under the frozen fixtures.",
        },
        {
            "lead_id": "PL03_FULL3D_ENVIRONMENTAL_NONLOCALITY",
            "state": "THEORY_SIGNATURE_ONLY",
            "exact_claim": "A conservative 3-D environmental field can differ from a local radial multiplier at saddles, mergers, and in an external field.",
            "closest_published_neighbor": "AQUAL/QUMOND external-field effects and refracted-gravity permittivity",
            "unique_discriminator": "Same-source 3-D morphology and external-field response with a distinct nonlocal scale and no curl artifact.",
            "counterexample": "No current real object has the required response-blind full-3-D source reconstruction.",
            "empirical_grade": "UNTESTED_SOURCE_BLOCKED",
            "theory_grade": "CONDITIONALLY_HEALTHY",
            "next_falsifier": "The full 3-D field is prediction-equivalent to a published comparator on every admitted fixture and source geometry.",
        },
    ]


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bindings = _validate_bindings(config)
    coverage = _coverage(config)
    leads = _publication_leads(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_EXACT_RECOVERY_COVERAGE_ACTIVE",
        "bindings": bindings,
        "coverage": coverage,
        "historical_workstreams": config["historical_recovery_workstreams"],
        "publication_leads": leads,
        "dual_grade_policy": config["dual_grade_policy"],
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
        "artifact_bindings": {
            "config_path": CONFIG_PATH.as_posix(),
            "config_sha256": file_sha256(CONFIG_PATH),
            "module_path": MODULE_PATH.as_posix(),
            "module_sha256": file_sha256(MODULE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_sha256": file_sha256(TEST_PATH),
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    expected = build_receipt()
    _require(dict(payload) == expected, "recovery receipt differs from deterministic rebuild")


def write_receipt() -> str:
    payload = build_receipt()
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        _require(OUTPUT_PATH.read_bytes() == encoded, "existing recovery receipt differs")
        return "EXISTING_IDENTICAL"
    with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, OUTPUT_PATH)
    except FileExistsError:
        _require(OUTPUT_PATH.read_bytes() == encoded, "concurrent recovery receipt differs")
        return "EXISTING_IDENTICAL"
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def validate_receipt() -> None:
    _require(OUTPUT_PATH.is_file(), "recovery receipt absent")
    payload = _read_json(OUTPUT_PATH, "recovery receipt")
    _require(type(payload) is dict, "receipt is not an object")
    validate_receipt_payload(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "concepts_recovered": receipt["coverage"]["concepts"],
                    "historical_workstreams": len(receipt["historical_workstreams"]),
                    "publication_leads": len(receipt["publication_leads"]),
                    "raw_scientific_rows": receipt["access_accounting"]["raw_scientific_rows"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
