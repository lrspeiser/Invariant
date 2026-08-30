"""Deterministic theory-gate routing for every open-gravity mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler import open_gravity_source_availability_contract_v2 as source_v2

CONFIG_PATH = Path("configs/open_gravity_theory_gate_matrix_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_theory_gate_matrix_v1.py")
TEST_PATH = Path("tests/test_open_gravity_theory_gate_matrix_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-theory-gate-matrix-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_theory_gate_matrix_v1.json")
_CANONICAL_MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_theory_gate_matrix_v1.py")
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_theory_gate_matrix_v1.py")
_CANONICAL_OUTPUT_PATH = Path("runs/gravity/open-gravity-theory-gate-matrix-v1/receipt.json")
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE_SHA256 = "2a685edb82163cc56dc90c492c55148b1377c8ca8e66394e18f2d22f67c3ed3c"
_CONFIG_CONTENT_SHA256 = "e5eb0d94a363dc5a89e527eeb84fcf127b647016af776171c5c6ad7c05ff71f9"
_SCHEMA = "invariant-open-gravity-theory-gate-matrix-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-theory-gate-matrix-receipt-1.0"

_DYNAMICAL_GATES = {
    "TG09_ACTION_CONSERVATION",
    "TG10_DOF_CONSTRAINTS",
    "TG11_PRINCIPAL_SYMBOL",
    "TG12_GHOST_GRADIENT",
    "TG13_HAMILTONIAN_ENERGY",
    "TG14_CAUSAL_COMMON_CONE",
    "TG15_RADIATION_POLARIZATION",
    "TG16_STRONG_COUPLING_CUTOFF",
}
_PRECISION_GATES = {"TG21_SOLAR_PPN", "TG22_PULSAR_BINARY", "TG23_COSMOLOGY"}
_QUANTUM_IDS = {"QG03", "QG04", "QG07", "QG08", "QG10", "QG11", "QG12"}
_STATIC_CLASSES = {"STATIC_FIELD", "SPATIAL_NONLOCAL"}
_NEXT_ARTIFACT = {
    "TG01_DIMENSIONS_LIMITS": "dimension-and-limit-proof",
    "TG02_FIELD_STATE": "typed-field-state-card",
    "TG03_SOURCE_COUPLING": "same-action-source-coupling",
    "TG04_EQUATIONS_OPERATOR": "complete-equation-or-operator-package",
    "TG05_BOUNDARY_INITIAL_GAUGE": "boundary-initial-gauge-contract",
    "TG06_SYNTHETIC_FIXTURES": "claim-matched-synthetic-suite",
    "TG07_FULL_3D_SOLVER": "independent-full-3d-or-spacetime-solver",
    "TG08_SYMMETRY_COVARIANCE": "covariance-and-symmetry-regressions",
    "TG09_ACTION_CONSERVATION": "action-or-exchange-ledger",
    "TG10_DOF_CONSTRAINTS": "constraint-and-degree-of-freedom-count",
    "TG11_PRINCIPAL_SYMBOL": "principal-symbol-derivation",
    "TG12_GHOST_GRADIENT": "kinetic-and-gradient-health-suite",
    "TG13_HAMILTONIAN_ENERGY": "hamiltonian-or-energy-bound",
    "TG14_CAUSAL_COMMON_CONE": "common-cauchy-and-cone-analysis",
    "TG15_RADIATION_POLARIZATION": "radiation-speed-polarization-loss-closure",
    "TG16_STRONG_COUPLING_CUTOFF": "strong-coupling-and-eft-cutoff",
    "TG17_QUANTUM_UNITARITY": "quantum-state-observable-unitarity-classical-limit",
    "TG18_MATTER_CAPTURE": "matter-motion-and-dissipation-free-capture-closure",
    "TG19_PHOTON_LENSING": "photon-null-lensing-closure",
    "TG20_CLOCK_REDSHIFT": "clock-path-cosmological-redshift-separation",
    "TG21_SOLAR_PPN": "solar-background-and-ppn-suite",
    "TG22_PULSAR_BINARY": "strong-field-binary-timing-suite",
    "TG23_COSMOLOGY": "background-and-perturbation-cosmology",
    "TG24_REAL_3D_SOURCE": "response-blind-real-3d-source-map",
    "TG25_REAL_DATA_CAMPAIGN": "target-blind-authorized-campaign",
}


class TheoryGateError(RuntimeError):
    """Raised whenever a theory-gate route fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TheoryGateError(message)


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
        raise TheoryGateError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TheoryGateError("committed binding unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "gate_vocabulary",
        "status_vocabulary",
        "ontology_classes",
        "route_rules",
        "matrix_contract",
        "access_contract",
        "claim_boundary",
        "output_path",
        "section_sha256",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == "open-gravity-theory-gate-matrix-v1", "ID changed")
    _require(config["status"] == "FROZEN_GATE_ROUTING_ZERO_RESPONSE_ACCESS", "status changed")
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    sealed = expected - {
        "schema",
        "package_id",
        "status",
        "purpose",
        "output_path",
        "section_sha256",
    }
    seals = config["section_sha256"]
    _require(type(seals) is dict and set(seals) == sealed, "section seals changed")
    for key in sealed:
        _require(seals[key] == content_sha256(config[key]), f"section {key} changed")
    gates = config["gate_vocabulary"]
    _require(len(gates) == 25, "gate count changed")
    _require([row["id"] for row in gates] == list(_NEXT_ARTIFACT), "gate order changed")
    matrix = config["matrix_contract"]
    _require(matrix["mechanisms"] == 420, "mechanism count changed")
    _require(matrix["gates"] == 25, "matrix gate count changed")
    _require(matrix["expected_rows"] == 10_500, "matrix rows changed")
    _require(matrix["observational_passes_allowed"] == 0, "observational authority changed")
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_FILE_SHA256, "config bytes changed")
    config = _read_json(path, "theory gate config")
    validate_config(config)
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config content changed")
    for binding in config["bindings"]:
        for artifact in binding["artifacts"]:
            expected = artifact["sha256"]
            _require(
                hashlib.sha256(_git_show(binding["commit"], artifact["path"])).hexdigest()
                == expected,
                f"committed {binding['role']} changed",
            )
            _require(
                file_sha256(_ROOT / artifact["path"]) == expected,
                f"working {binding['role']} changed",
            )
    _source_config, predecessor = source_v2.load_inputs()
    catalog = source_v2.mechanism_catalog(predecessor)
    _require(len(catalog) == 420, "catalog count changed")
    return config, predecessor, catalog


def ontology_class(mechanism: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    mechanism_id = mechanism["mechanism_id"]
    architecture = mechanism["architecture"]
    for name, members in config["ontology_classes"].items():
        if mechanism_id in members or architecture in members:
            return name
    raise TheoryGateError(f"unrouted mechanism: {mechanism_id}")


def _required_for_current_claim(mechanism: Mapping[str, Any], category: str, gate_id: str) -> bool:
    mechanism_id = mechanism["mechanism_id"]
    if gate_id in _DYNAMICAL_GATES:
        return category not in _STATIC_CLASSES or mechanism_id == "GP01-AQUAL"
    if gate_id == "TG17_QUANTUM_UNITARITY":
        return mechanism_id in _QUANTUM_IDS
    if gate_id in _PRECISION_GATES:
        return category in {
            "SPACETIME_COUPLED",
            "HISTORY_WAVE_FEEDBACK",
            "STOCHASTIC",
            "GRAVITY_LIGHT_ONTOLOGY",
        }
    return True


def _evidence_status(
    mechanism: Mapping[str, Any], category: str, gate_id: str, required: bool
) -> str:
    mechanism_id = mechanism["mechanism_id"]
    family = mechanism["mechanism_family"]
    if mechanism_id == "GP01-ACTION_PLACEHOLDER":
        return "INCOMPLETE_QUARANTINE"
    if not required:
        return "NOT_APPLICABLE_CURRENT_SCOPE"
    if gate_id == "TG24_REAL_3D_SOURCE":
        return "BLOCKED_MISSING_SOURCE"
    if gate_id == "TG25_REAL_DATA_CAMPAIGN":
        return "BLOCKED_UPSTREAM_GATES"
    if gate_id == "TG01_DIMENSIONS_LIMITS":
        return "PASS_REGISTERED" if family != "GRAVITY_LIGHT_ONTOLOGY" else "PARTIAL"
    if mechanism_id == "GP01-AQUAL":
        if gate_id in {
            "TG02_FIELD_STATE",
            "TG03_SOURCE_COUPLING",
            "TG04_EQUATIONS_OPERATOR",
            "TG09_ACTION_CONSERVATION",
            "TG18_MATTER_CAPTURE",
        }:
            return "PASS_PRIMARY_SOURCE_NOT_INDEPENDENTLY_REDERIVED"
        if gate_id in {
            "TG05_BOUNDARY_INITIAL_GAUGE",
            "TG06_SYNTHETIC_FIXTURES",
            "TG07_FULL_3D_SOLVER",
            "TG08_SYMMETRY_COVARIANCE",
        }:
            return "PASS_TARGET_FREE"
        return "BLOCKED_MISSING_DEFINITION"
    if mechanism_id in {"QG01", "QG02"} and gate_id in {
        "TG02_FIELD_STATE",
        "TG03_SOURCE_COUPLING",
        "TG04_EQUATIONS_OPERATOR",
        "TG08_SYMMETRY_COVARIANCE",
        "TG09_ACTION_CONSERVATION",
        "TG10_DOF_CONSTRAINTS",
        "TG11_PRINCIPAL_SYMBOL",
        "TG12_GHOST_GRADIENT",
        "TG13_HAMILTONIAN_ENERGY",
        "TG14_CAUSAL_COMMON_CONE",
        "TG15_RADIATION_POLARIZATION",
        "TG18_MATTER_CAPTURE",
        "TG19_PHOTON_LENSING",
        "TG20_CLOCK_REDSHIFT",
        "TG21_SOLAR_PPN",
        "TG22_PULSAR_BINARY",
        "TG23_COSMOLOGY",
    }:
        return "PASS_PRIMARY_SOURCE_NOT_INDEPENDENTLY_REDERIVED"
    if family.startswith("TWELL"):
        if gate_id in {
            "TG03_SOURCE_COUPLING",
            "TG04_EQUATIONS_OPERATOR",
            "TG05_BOUNDARY_INITIAL_GAUGE",
            "TG06_SYNTHETIC_FIXTURES",
            "TG08_SYMMETRY_COVARIANCE",
            "TG18_MATTER_CAPTURE",
            "TG20_CLOCK_REDSHIFT",
        }:
            return "PARTIAL"
        if gate_id == "TG07_FULL_3D_SOLVER":
            return "BLOCKED_MISSING_SOLVER"
        return "BLOCKED_MISSING_DEFINITION"
    if family == "GP01":
        if gate_id in {
            "TG02_FIELD_STATE",
            "TG03_SOURCE_COUPLING",
            "TG04_EQUATIONS_OPERATOR",
            "TG05_BOUNDARY_INITIAL_GAUGE",
            "TG06_SYNTHETIC_FIXTURES",
            "TG18_MATTER_CAPTURE",
            "TG20_CLOCK_REDSHIFT",
        }:
            return "PARTIAL"
        if gate_id == "TG07_FULL_3D_SOLVER":
            return "BLOCKED_MISSING_SOLVER"
        return "BLOCKED_MISSING_DEFINITION"
    if category == "GRAVITY_LIGHT_ONTOLOGY":
        return (
            "REQUIRED_UNRUN"
            if gate_id == "TG17_QUANTUM_UNITARITY"
            else "BLOCKED_MISSING_DEFINITION"
        )
    raise TheoryGateError("evidence route missing")


def iter_gate_rows(
    config: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]]
) -> Iterable[dict[str, Any]]:
    for mechanism in catalog:
        category = ontology_class(mechanism, config)
        for gate in config["gate_vocabulary"]:
            gate_id = gate["id"]
            required = _required_for_current_claim(mechanism, category, gate_id)
            status = _evidence_status(mechanism, category, gate_id, required)
            contract = {
                "mechanism_id": mechanism["mechanism_id"],
                "ontology_class": category,
                "gate_id": gate_id,
                "required_for_current_claim": required,
                "evidence_status": status,
                "next_artifact": _NEXT_ARTIFACT[gate_id],
            }
            yield {
                "mechanism_id": mechanism["mechanism_id"],
                "mechanism_family": mechanism["mechanism_family"],
                "discovery_lane": mechanism["discovery_lane"],
                "architecture": mechanism["architecture"],
                "ontology_class": category,
                "gate_id": gate_id,
                "required_for_current_claim": required,
                "evidence_status": status,
                "next_artifact": _NEXT_ARTIFACT[gate_id],
                "gate_contract_sha256": content_sha256(contract),
            }


def _stream_root(rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(_canonical(row))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def build_receipt() -> dict[str, Any]:
    config, _predecessor, catalog = load_inputs()
    rows = list(iter_gate_rows(config, catalog))
    count, root = _stream_root(rows)
    _require(count == 10_500, "gate matrix row count changed")
    status_counts = Counter(row["evidence_status"] for row in rows)
    category_counts = Counter(row["ontology_class"] for row in rows)
    gate_counts = {
        gate["id"]: dict(
            sorted(
                Counter(
                    row["evidence_status"] for row in rows if row["gate_id"] == gate["id"]
                ).items()
            )
        )
        for gate in config["gate_vocabulary"]
    }
    _require(status_counts.get("PASS_OBSERVATIONAL", 0) == 0, "observational pass appeared")
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_COMPLETE_THEORY_GATE_ROUTING_ZERO_OBSERVATIONAL_AUTHORITY",
        "access_accounting": config["access_contract"],
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
            "predecessors": config["bindings"],
        },
        "matrix": {
            "rows": count,
            "stream_sha256": root,
            "mechanisms": len(catalog),
            "gates": len(config["gate_vocabulary"]),
            "status_counts": dict(sorted(status_counts.items())),
            "ontology_class_row_counts": dict(sorted(category_counts.items())),
            "gate_status_counts": gate_counts,
            "observational_passes": 0,
        },
        "gate_vocabulary": config["gate_vocabulary"],
        "route_rules": config["route_rules"],
        "claim_boundary": config["claim_boundary"],
        "section_sha256": config["section_sha256"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _output_path() -> Path:
    return _path(OUTPUT_PATH, _CANONICAL_OUTPUT_PATH, "output")


def write_receipt() -> str:
    path = _output_path()
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(path.read_bytes() == payload, "refusing to overwrite nonidentical receipt")
        return "EXISTING_IDENTICAL"
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


def validate_receipt_payload(stored: Mapping[str, Any]) -> None:
    body = {key: value for key, value in stored.items() if key != "content_sha256"}
    _require(stored.get("content_sha256") == content_sha256(body), "self-hash changed")
    _require(dict(stored) == build_receipt(), "receipt is not reproducible")


def validate_receipt() -> None:
    validate_receipt_payload(_read_json(_output_path(), "theory gate receipt"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("summary", "build", "validate"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summary":
        receipt = build_receipt()
        print(json.dumps(receipt["matrix"], sort_keys=True, indent=2))
    elif args.command == "build":
        print(write_receipt())
    else:
        validate_receipt()
        print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
