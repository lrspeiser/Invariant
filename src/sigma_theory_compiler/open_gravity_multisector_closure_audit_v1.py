"""Exact target-free audit of matter, light, clocks, waves, capture, and cosmology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_multisector_closure_audit_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_multisector_closure_audit_v1.py")
TEST_PATH = Path("tests/test_open_gravity_multisector_closure_audit_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-multisector-closure-audit-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_multisector_closure_audit_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_multisector_closure_audit_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_multisector_closure_audit_v1.py")
_CANONICAL_OUTPUT_PATH = Path("runs/gravity/open-gravity-multisector-closure-audit-v1/receipt.json")
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "5f00130373066a4a8768c04601512aee74dfe7de5a8ddf0970094015f8764424"
_CONFIG_CONTENT_SHA256 = "8bf0f0aea5e3b41553d67914ffe9d5d04165c96ccfa86eeb2f786a86e0c91bfb"
_SCHEMA = "invariant-open-gravity-multisector-closure-audit-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-multisector-closure-audit-receipt-1.0"
_SECTORS = (
    "MATTER",
    "PHOTON_AND_LENSING",
    "CLOCK",
    "REDSHIFT",
    "GRAVITATIONAL_WAVES",
    "CAPTURE_AND_CLUMPING",
    "SOLAR_AND_LAB",
    "PULSAR_AND_COMPACT_BINARY",
    "FLRW_AND_COSMOLOGY",
    "HAMILTONIAN_STABILITY_AND_CONSTRAINTS",
    "QUANTUM_GRAVITY",
)


class ClosureAuditError(RuntimeError):
    """Raised when a multisector claim or seal fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureAuditError(message)


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


def _path(current: Path, expected: Path, label: str) -> Path:
    _require(current == expected, f"canonical {label} path changed")
    path = (_ROOT / expected).resolve()
    _require(path.is_relative_to(_ROOT), f"{label} escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureAuditError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=_ROOT, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ClosureAuditError("committed binding unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "evidence_packages",
        "new_bindings",
        "sector_ledger",
        "required_gates",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == "open-gravity-multisector-closure-audit-v1", "ID changed")
    _require(
        config["status"] == "FROZEN_TARGET_FREE_MULTI_SECTOR_PARTIAL_AND_BLOCKED_LEDGER",
        "status changed",
    )
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(tuple(row["sector"] for row in config["sector_ledger"]) == _SECTORS, "sectors changed")
    _require(len(config["evidence_packages"]) == 7, "evidence count changed")
    _require(len(config["required_gates"]) == 12, "gate count changed")
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def _verify_blob(commit: str, path: str, expected: str, role: str) -> None:
    _require(
        hashlib.sha256(_git_show(commit, path)).hexdigest() == expected, f"committed {role} changed"
    )
    _require(file_sha256(_ROOT / path) == expected, f"working {role} changed")


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "multisector config")
    validate_config(config)
    for package in config["evidence_packages"]:
        _verify_blob(
            package["commit"], package["config_path"], package["config_sha256"], package["id"]
        )
        _verify_blob(
            package["commit"], package["receipt_path"], package["receipt_sha256"], package["id"]
        )
    for binding in config["new_bindings"]:
        _verify_blob(binding["commit"], binding["path"], binding["sha256"], binding["role"])
    return config


def load_evidence(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for package in config["evidence_packages"]:
        path = (_ROOT / package["receipt_path"]).resolve()
        _require(path.is_relative_to(_ROOT), "evidence path escaped repository")
        _require(
            file_sha256(path) == package["receipt_sha256"], "evidence bytes changed before read"
        )
        receipt = _read_json(path, package["id"])
        _require(receipt["decision"] == package["expected_decision"], "evidence decision changed")
        evidence[package["id"]] = receipt
    for binding in config["new_bindings"]:
        path = (_ROOT / binding["path"]).resolve()
        _require(path.is_relative_to(_ROOT), "new binding escaped repository")
        _require(file_sha256(path) == binding["sha256"], "new binding changed before read")
        evidence[binding["role"]] = _read_json(path, binding["role"])
    return evidence


def _all_zero(value: Any) -> bool:
    if type(value) is dict:
        return all(_all_zero(item) for item in value.values())
    return type(value) in (int, float) and value == 0


def _gate(passed: bool, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "metrics": dict(metrics)}


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    evidence = load_evidence(config)
    sectors = {row["sector"]: row for row in config["sector_ledger"]}
    gates: dict[str, dict[str, Any]] = {}
    gates["EVIDENCE_BYTES_AND_COMMITS_EXACT"] = _gate(
        len(evidence) == 10,
        {"legacy_theory_packages": 7, "new_exact_bindings": 3},
    )

    matter = evidence["UNIVERSAL_CONFORMAL_SOURCE"]["claim_boundary"]
    gates["UNIVERSAL_MATTER_SOURCE_IDENTITY_ONLY"] = _gate(
        matter["universal_conformal_source_identity_established"] is True
        and matter["physical_source_profile_established"] is False
        and matter["on_shell_solution_established"] is False,
        {
            "source_identity": True,
            "physical_profile": False,
            "on_shell_background": False,
            "sector_status": sectors["MATTER"]["status"],
        },
    )

    gates["PHOTON_CONFORMAL_CONE_WITH_LENSING_BLOCK"] = _gate(
        matter["lensing_success_established"] is False
        and sectors["PHOTON_AND_LENSING"]["status"] == "PARTIAL_COMMON_CONFORMAL_NULL_CONE_ONLY",
        {
            "local_conformal_null_cone": True,
            "metric_backreaction": False,
            "lensing_prediction": False,
        },
    )

    clock_gate = evidence["SYNTHETIC_UNIVERSE"]["suite"]["gates"]["CLOCK_ENDPOINT_NOT_PATH_RULE"]
    gates["CLOCK_ENDPOINT_RULE_WITH_DYNAMIC_REDSHIFT_BLOCK"] = _gate(
        clock_gate["passed"] is True
        and clock_gate["metrics"]["path_spread"] == 0.0
        and sectors["REDSHIFT"]["status"]
        == "BLOCKED_NO_COMPLETE_DYNAMIC_METRIC_AND_SOURCE_HISTORY",
        {
            "static_endpoint_rule": True,
            "extra_static_path_accumulation": False,
            "dynamic_redshift_completion": False,
        },
    )

    solar_gw = evidence["SOLAR_GW_NECESSARY"]["claim_boundary"]
    gates["GW_NECESSARY_CONE_BOUND_WITHOUT_OBSERVATIONAL_PASS"] = _gate(
        solar_gw["restricted_necessary_conditions_established"] is True
        and solar_gw["gw_speed_observational_pass_established"] is False
        and solar_gw["disformal_branch_viable"] is False,
        {
            "restricted_conditions": True,
            "gw_observational_pass": False,
            "disformal_viability": False,
        },
    )

    gates["CAPTURE_REQUIRES_ENERGY_ENTROPY_RECEIVER"] = _gate(
        sectors["CAPTURE_AND_CLUMPING"]["status"]
        == "BLOCKED_CONSERVATIVE_SYSTEM_HAS_NO_DISSIPATIVE_RECEIVER",
        {
            "deeper_conservative_binding_possible": True,
            "irreversible_capture_derived": False,
            "receiver_defined": False,
        },
    )

    gates["SOLAR_AND_PULSAR_PHYSICAL_GATES_BLOCKED"] = _gate(
        solar_gw["solar_viability_established"] is False
        and solar_gw["on_shell_background_established"] is False
        and sectors["PULSAR_AND_COMPACT_BINARY"]["status"].startswith("BLOCKED_"),
        {
            "solar_viability": False,
            "regular_solar_background": False,
            "binary_radiation_prediction": False,
        },
    )

    flrw = evidence["FLRW_BACKGROUND"]["claim_boundary"]
    gates["FLRW_EQUATIONS_WITH_HISTORY_OBSTRUCTION"] = _gate(
        flrw["restricted_flat_flrw_equations_established"] is True
        and flrw["healthy_history_established"] is False
        and flrw["perturbation_stability_established"] is False,
        {"background_equations": True, "healthy_history": False, "stable_perturbations": False},
    )

    principal = evidence["EXTERNAL_METRIC_PRINCIPAL_SYMBOL"]["claim_boundary"]
    hamiltonian = evidence["SCALAR_HAMILTONIAN"]["claim_boundary"]
    adm = evidence["ADM_CONSTRAINT_PROPAGATION"]["claim_boundary"]
    gates["HAMILTONIAN_PRINCIPAL_AND_CONSTRAINT_PARTIALS"] = _gate(
        principal["full_H3_passed"] is False
        and principal["full_H4_passed"] is False
        and hamiltonian["necessary_legendre_and_slice_health_conditions_derived"] is True
        and hamiltonian["physical_hamiltonian_positivity_established"] is False
        and adm["CP11_3_complete"] is True
        and adm["closed_healthy_theory_established"] is False,
        {
            "external_metric_principal_symbol_partial": True,
            "full_hyperbolicity": False,
            "restricted_hamiltonian": True,
            "physical_hamiltonian_positivity": False,
            "conditional_adm_constraint_identity": True,
        },
    )

    gp_receipt = evidence["GP01_FULL3D"]
    gates["GP01_HAS_NO_COMMON_MULTI_SECTOR_ACTION"] = _gate(
        gp_receipt["status"] == "PASS_FULL3D_ELLIPTIC_PARTIAL_TELEGRAPH_TARGET_FREE_ONLY"
        and "covariant action health" in gp_receipt["claim_boundary"]["does_not_establish"]
        and "lensing or redshift closure" in gp_receipt["claim_boundary"]["does_not_establish"]
        and sectors["PHOTON_AND_LENSING"]["status"].startswith("PARTIAL_")
        and sectors["QUANTUM_GRAVITY"]["status"].startswith("BLOCKED_"),
        {
            "static_field_mechanics": True,
            "common_multisector_action": False,
            "observational_closure": False,
        },
    )

    quantum_receipt = evidence["GRAVITY_LIGHT_QUANTUM_CARDS"]
    quantum_blocks = quantum_receipt["claim_boundary"]["does_not_establish"]
    gates["QUANTUM_CLAIMS_BLOCKED"] = _gate(
        "quantization of observed gravity" in quantum_blocks
        and "a graviton detection" in quantum_blocks
        and sectors["QUANTUM_GRAVITY"]["status"]
        == "BLOCKED_NO_QUANTIZATION_OR_NONCLASSICAL_OBSERVABLE",
        {"typed_quantum_cards": 13, "quantized_gp01": False, "nonclassical_witness": False},
    )

    zero_evidence = all(
        _all_zero(receipt["zero_access_and_compute"])
        for key, receipt in evidence.items()
        if key in {row["id"] for row in config["evidence_packages"]}
    )
    gates["ZERO_RESPONSE_ACCESS"] = _gate(
        zero_evidence and all(value == 0 for value in config["access_contract"].values()),
        {"theory_packages_zero_access": zero_evidence, **config["access_contract"]},
    )

    _require(list(gates) == config["required_gates"], "gate order changed")
    _require(all(row["passed"] is True for row in gates.values()), "multisector gate failed")
    return {
        "sectors": len(sectors),
        "status_counts": {
            "PARTIAL": sum(row["status"].startswith("PARTIAL_") for row in sectors.values()),
            "BLOCKED": sum(row["status"].startswith("BLOCKED_") for row in sectors.values()),
        },
        "gates": gates,
        "passed": len(gates),
        "failed": 0,
        "sector_ledger": config["sector_ledger"],
        "observational_authority": False,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_MULTI_SECTOR_PARTIAL_AND_BLOCKED_LEDGER_TARGET_FREE_ONLY",
        "bindings": {
            "config": {
                "path": _CANONICAL_CONFIG_PATH.as_posix(),
                "sha256": file_sha256(_ROOT / _CANONICAL_CONFIG_PATH),
                "content_sha256": content_sha256(config),
            },
            "module": {
                "path": _CANONICAL_MODULE_PATH.as_posix(),
                "sha256": file_sha256(module_path),
            },
            "test": {"path": _CANONICAL_TEST_PATH.as_posix(), "sha256": file_sha256(test_path)},
            "evidence_packages": config["evidence_packages"],
            "new_bindings": config["new_bindings"],
        },
        "suite": run_suite(config),
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "receipt is not an object")
    _require(payload == build_receipt(), "receipt is not reproducible")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload["content_sha256"] == content_sha256(body), "receipt self-hash changed")


def _output_path() -> Path:
    return _path(OUTPUT_PATH, _CANONICAL_OUTPUT_PATH, "output")


def write_receipt() -> str:
    path = _output_path()
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "CREATED"


def validate_receipt() -> None:
    validate_receipt_payload(_read_json(_output_path(), "multisector receipt"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
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
                    "sectors": receipt["suite"]["sectors"],
                    "status_counts": receipt["suite"]["status_counts"],
                    "observational_authority": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
