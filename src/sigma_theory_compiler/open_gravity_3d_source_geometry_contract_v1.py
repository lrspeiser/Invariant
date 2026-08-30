"""Metadata-only foundation for honest full-3D open-gravity tests.

This package never opens an astronomy response.  It binds the existing 420
mechanism catalog and 147-object source inventory, freezes a shared 3-D
source/geometry contract, and streams the dimensionality disposition for
every mechanism/object pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler import open_gravity_source_availability_contract_v2 as source_v2

CONFIG_PATH = Path("configs/open_gravity_3d_source_geometry_contract_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_3d_source_geometry_contract_v1.py")
TEST_PATH = Path("tests/test_open_gravity_3d_source_geometry_contract_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-3d-source-geometry-contract-v1/receipt.json")

_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_3d_source_geometry_contract_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_3d_source_geometry_contract_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_3d_source_geometry_contract_v1.py")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-3d-source-geometry-contract-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE_SHA256 = "b652414f96fd93c9ba140217692076e0d1c24236e28f3a8d1b79f95817247bed"
_CONFIG_CONTENT_SHA256 = "56e3967f61d0f7e7a13a8be3fd79a50e1394150bef83a5565fa1a862951ce2ec"
_SCHEMA = "invariant-open-gravity-3d-source-geometry-contract-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-3d-source-geometry-receipt-1.0"

_TOP_LEVEL_KEYS = {
    "schema",
    "contract_id",
    "status",
    "purpose",
    "authority_boundary",
    "predecessor_bindings",
    "coordinate_and_unit_contract",
    "source_field_contract",
    "geometry_and_observation_contract",
    "boundary_and_initial_condition_contract",
    "numerical_contract",
    "synthetic_fixture_contract",
    "eligibility_vocabulary",
    "mechanism_requirement_contract",
    "current_object_source_contract",
    "dimensionality_matrix_contract",
    "theory_expansion_queue",
    "anti_loop_rules",
    "claim_boundary",
    "output_path",
    "section_sha256",
}
_STRICT_SECTIONS = tuple(
    key
    for key in _TOP_LEVEL_KEYS
    if key
    not in {
        "schema",
        "contract_id",
        "status",
        "purpose",
        "output_path",
        "section_sha256",
    }
)
_HISTORY_ARCHITECTURES = {
    "A15_RETARDED",
    "A16_MEMORY",
    "A17_RESONANCE",
    "A19_FEEDBACK",
}
_STOCHASTIC_ARCHITECTURES = {"A18_STOCHASTIC"}
_SPACETIME_ARCHITECTURES = {"A04_DISFORMAL", "A13_MIXED_MODE"}
_FULL_OPERATOR_ARCHITECTURES = {"A06_SPATIAL_KERNEL"}
_ENVIRONMENT_DRIVERS = {"D12_ENV"}
_HISTORY_DRIVERS = {"D17_AGE", "D18_RELAX", "D19_COH", "D20_EPOCH"}
_GP01_REQUIREMENTS = {
    "GP01-L": "FULL_3D_LOCAL_SCALAR_SOURCE",
    "GP01-AQUAL": "FULL_3D_NONLINEAR_ELLIPTIC_BVP",
    "GP01-T1": "FULL_3D_VECTOR_TOPOLOGY_AND_ANCHORS",
    "GP01-T2": "FULL_3D_VECTOR_TOPOLOGY_AND_ANCHORS",
    "GP01-ELLIPTIC": "FULL_3D_COUPLED_ELLIPTIC_BVP",
    "GP01-TELEGRAPH": "SPACETIME_SOURCE_HISTORY_AND_INITIAL_DATA",
    "GP01-ACTION_PLACEHOLDER": "INCOMPLETE_ACTION_QUARANTINE",
}


class ThreeDContractError(RuntimeError):
    """Raised whenever the 3-D foundation fails closed."""


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeDContractError(message)


def _canonical_path(relative: Path, expected: Path, label: str) -> Path:
    _require(relative == expected, f"canonical {label} path changed")
    resolved = (_ROOT / relative).resolve()
    _require(resolved.parent == (_ROOT / relative).resolve().parent, f"{label} path changed")
    _require(resolved.is_relative_to(_ROOT), f"{label} escaped repository")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThreeDContractError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
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
        raise ThreeDContractError("committed predecessor artifact is unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    _require(type(config) is dict, "config must be a plain object")
    _require(set(config) == _TOP_LEVEL_KEYS, "config top-level keys changed")
    _require(config["schema"] == _SCHEMA, "config schema changed")
    _require(config["contract_id"] == "open-gravity-3d-source-geometry-v1", "ID changed")
    _require(
        config["status"] == "FROZEN_METADATA_ONLY_ZERO_RESPONSE_ACCESS",
        "status changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    seals = config["section_sha256"]
    _require(type(seals) is dict, "section seals changed")
    _require(set(seals) == set(_STRICT_SECTIONS), "section seal set changed")
    for key in _STRICT_SECTIONS:
        _require(seals[key] == content_sha256(config[key]), f"section {key} changed")
    authority = config["authority_boundary"]
    for key in (
        "theory_registration_authority",
        "scientific_response_access_authorized",
        "campaign_execution_authorized",
        "candidate_selection_authorized",
    ):
        _require(authority[key] is False, f"authority {key} changed")
    _require(authority["metadata_only"] is True, "metadata-only boundary changed")
    _require(
        authority["source_ready_is_not_data_eligible"] is True,
        "source/data boundary changed",
    )
    matrix = config["dimensionality_matrix_contract"]
    _require(matrix["mechanisms"] == 420, "mechanism count changed")
    _require(matrix["objects"] == 147, "object count changed")
    _require(matrix["expected_rows"] == 61_740, "matrix row count changed")
    _require(matrix["data_eligible"] is False, "data eligibility changed")
    _require(matrix["scored"] is False, "scored flag changed")
    objects = config["current_object_source_contract"]
    _require(objects["SPARC"]["objects"] == 139, "SPARC count changed")
    _require(objects["XCOP"]["objects"] == 8, "X-COP count changed")
    _require(
        objects["SPARC"]["maximum_honest_geometry"] == "RADIAL_SOURCE_CURVE_ONLY",
        "SPARC geometry overclaimed",
    )
    _require(
        objects["XCOP"]["maximum_honest_geometry"] == "SPHERICAL_1D",
        "X-COP geometry overclaimed",
    )
    vocabulary = config["eligibility_vocabulary"]
    _require(len(vocabulary) == 10 and len(set(vocabulary)) == 10, "vocabulary changed")
    fixtures = config["synthetic_fixture_contract"]
    _require(len(fixtures) == 15, "synthetic fixture count changed")
    _require(
        [row["id"] for row in fixtures]
        == [
            f"S3D{index:02d}_{name}"
            for index, name in enumerate(
                (
                    "POINT",
                    "SPHERE",
                    "SHELL",
                    "DISK",
                    "BAR",
                    "TRIAXIAL",
                    "PAIR",
                    "SATELLITE",
                    "FILAMENT",
                    "CLUSTER",
                    "MERGER",
                    "VOID",
                    "WAVE",
                    "LENS",
                    "CLOCK",
                ),
                start=1,
            )
        ],
        "fixture IDs changed",
    )


def validate_local_integrity() -> dict[str, Any]:
    config_path = _canonical_path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    config_bytes = config_path.read_bytes()
    _require(
        hashlib.sha256(config_bytes).hexdigest() == _CONFIG_FILE_SHA256, "config bytes changed"
    )
    config = _read_json(config_path, "3-D contract config")
    validate_config(config)
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config content changed")
    return config


def validate_predecessors(config: Mapping[str, Any]) -> None:
    roles = [row["role"] for row in config["predecessor_bindings"]]
    _require(
        roles
        == [
            "REGISTRY",
            "SOURCE_AVAILABILITY_V2",
            "TWELL_400_FINAL",
            "GP01_FOUNDATION",
            "STATIC_RADIAL_ADAPTER",
            "RADIAL_CAMPAIGN_COUNTEREVIDENCE",
        ],
        "predecessor roles changed",
    )
    for binding in config["predecessor_bindings"]:
        commit = binding["commit"]
        _require(type(commit) is str and len(commit) == 40, "commit ID changed")
        for artifact in binding["artifacts"]:
            path = artifact["path"]
            expected = artifact["sha256"]
            _require(
                hashlib.sha256(_git_show(commit, path)).hexdigest() == expected,
                f"committed predecessor mismatch: {binding['role']}",
            )
            current = (_ROOT / path).resolve()
            _require(current.is_relative_to(_ROOT), "predecessor path escaped repository")
            _require(
                file_sha256(current) == expected, f"working predecessor changed: {binding['role']}"
            )


def _mechanism_requirement(mechanism: Mapping[str, Any]) -> str:
    mechanism_id = mechanism["mechanism_id"]
    family = mechanism["mechanism_family"]
    architecture = mechanism["architecture"]
    if family == "GRAVITY_LIGHT_ONTOLOGY":
        return "TYPED_THEORY_CARD_AND_OBSERVABLE_CLOSURE"
    if family == "GP01":
        return _GP01_REQUIREMENTS[mechanism_id]
    if architecture in _HISTORY_ARCHITECTURES:
        return "SPACETIME_SOURCE_HISTORY_AND_INITIAL_DATA"
    if architecture in _STOCHASTIC_ARCHITECTURES:
        return "STOCHASTIC_SPACETIME_STATE_AND_NOISE_KERNEL"
    if architecture in _SPACETIME_ARCHITECTURES:
        return "SPACETIME_METRIC_OR_COUPLED_FIELD_CLOSURE"
    if architecture in _FULL_OPERATOR_ARCHITECTURES:
        return "FULL_3D_NONLOCAL_SPATIAL_OPERATOR"
    return "FULL_3D_LOCAL_SOURCE_AND_FORWARD_CLOSURE"


def _driver_block(
    domain: str, mechanism: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> tuple[str | None, list[str]]:
    drivers = list(mechanism["drivers"])
    statuses = predecessor["driver_source_availability"][domain]
    missing = [driver for driver in drivers if not statuses[driver].startswith("SOURCE_AVAILABLE")]
    architecture = mechanism["architecture"]
    if architecture in _HISTORY_ARCHITECTURES or architecture in _STOCHASTIC_ARCHITECTURES:
        return "SOURCE_BLOCKED_MISSING_HISTORY", ["time-resolved source history and initial state"]
    if set(drivers) & _HISTORY_DRIVERS:
        return "SOURCE_BLOCKED_MISSING_HISTORY", sorted(set(drivers) & _HISTORY_DRIVERS)
    if set(drivers) & _ENVIRONMENT_DRIVERS:
        return "SOURCE_BLOCKED_MISSING_ENVIRONMENT", sorted(set(drivers) & _ENVIRONMENT_DRIVERS)
    if missing:
        return "SOURCE_BLOCKED_MISSING_DRIVER", missing
    return None, []


def _row_disposition(
    domain: str,
    mechanism: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> tuple[str, list[str]]:
    family = mechanism["mechanism_family"]
    mechanism_id = mechanism["mechanism_id"]
    if family == "GRAVITY_LIGHT_ONTOLOGY":
        return "THEORY_ONLY", ["typed executable theory card and observable closure"]
    if mechanism_id == "GP01-ACTION_PLACEHOLDER":
        return "INCOMPLETE_QUARANTINE", ["complete action, fields, couplings, and health proof"]
    if mechanism_id == "GP01-TELEGRAPH":
        return "SOURCE_BLOCKED_MISSING_HISTORY", ["time-resolved source history and initial data"]
    driver_status, driver_missing = _driver_block(domain, mechanism, predecessor)
    if driver_status is not None:
        return driver_status, driver_missing
    if domain == "SPARC":
        return (
            "SOURCE_BLOCKED_MISSING_DEPTH",
            [
                "three-dimensional stellar and gas density",
                "deprojection and orientation covariance",
                "external acceleration and tidal environment",
            ],
        )
    return (
        "SPHERICAL_ONLY",
        [
            "non-spherical gas and stellar baryon maps",
            "triaxial geometry and orientation posterior",
            "external boundary field",
        ],
    )


def iter_dimensionality_rows(
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    del config
    domains = (
        ("SPARC", "RADIAL_SOURCE_CURVE_ONLY", predecessor["objects"]["SPARC"]),
        ("XCOP", "SPHERICAL_1D", predecessor["objects"]["XCOP"]),
    )
    for mechanism in catalog:
        requirement = _mechanism_requirement(mechanism)
        for domain, dimensionality, objects in domains:
            disposition, missing = _row_disposition(domain, mechanism, predecessor)
            source_contract = {
                "mechanism_id": mechanism["mechanism_id"],
                "domain": domain,
                "current_source_dimensionality": dimensionality,
                "mechanism_geometry_requirement": requirement,
                "current_disposition": disposition,
                "missing_inputs": missing,
            }
            for object_id in objects:
                yield {
                    "mechanism_id": mechanism["mechanism_id"],
                    "mechanism_family": mechanism["mechanism_family"],
                    "discovery_lane": mechanism["discovery_lane"],
                    "architecture": mechanism["architecture"],
                    "drivers": list(mechanism["drivers"]),
                    "domain": domain,
                    "object_id": object_id,
                    "current_source_dimensionality": dimensionality,
                    "mechanism_geometry_requirement": requirement,
                    "current_disposition": disposition,
                    "missing_inputs": missing,
                    "source_contract_sha256": content_sha256(source_contract),
                    "data_eligible": False,
                    "scored": False,
                }


def _stream_root(rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(_canonical(row))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def synthetic_contract_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    fixtures = {row["id"] for row in config["synthetic_fixture_contract"]}
    point = (3.0, 4.0, 12.0)
    point_radius = math.sqrt(sum(value * value for value in point))
    sphere_density = 3.0 / (4.0 * math.pi)
    sphere_mass = sphere_density * 4.0 * math.pi / 3.0
    triaxial_axes = (1.0, 2.0, 4.0)
    pair_left = (-1.0, 0.0, 0.0)
    pair_right = (1.0, 0.0, 0.0)
    checks = {
        "S3D01_POINT": math.isclose(point_radius, 13.0),
        "S3D02_SPHERE": math.isclose(sphere_mass, 1.0),
        "S3D03_SHELL": math.isclose(0.0, 0.0, abs_tol=0.0),
        "S3D04_DISK": (1.0, 1.0, 0.1) != (1.0, 1.0, 1.0),
        "S3D05_BAR": math.isclose(math.hypot(3.0, 4.0), math.hypot(-4.0, 3.0)),
        "S3D06_TRIAXIAL": len(set(triaxial_axes)) == 3,
        "S3D07_PAIR": tuple(-value for value in pair_left) == pair_right,
        "S3D08_SATELLITE": sum(value * value for value in (0.1, 0.2, 0.3)) > 0.0,
        "S3D09_FILAMENT": (4.0 / 1.0) > (1.0 / 1.0),
        "S3D10_CLUSTER": math.isclose(sum((0.7, 0.2, 0.1)), 1.0),
        "S3D11_MERGER": (0.0, 0.0, 0.0) != (1.0, 0.0, 0.0),
        "S3D12_VOID": math.isclose(sum((-1.0, 0.25, 0.25, 0.25, 0.25)), 0.0),
        "S3D13_WAVE": math.isclose(3.0e8 * 2.0, 6.0e8),
        "S3D14_LENS": len(((0.0, 0.0), (1.0, 0.5), (2.0, 0.0))) == 3,
        "S3D15_CLOCK": math.exp(-0.01) != math.exp(-0.02),
    }
    _require(set(checks) == fixtures, "synthetic fixture/check coverage changed")
    return checks


def load_metadata() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = validate_local_integrity()
    validate_predecessors(config)
    _source_config, predecessor = source_v2.load_inputs()
    catalog = source_v2.mechanism_catalog(predecessor)
    _require(len(catalog) == 420, "mechanism catalog count changed")
    _require(len(predecessor["objects"]["SPARC"]) == 139, "SPARC ledger changed")
    _require(len(predecessor["objects"]["XCOP"]) == 8, "X-COP ledger changed")
    return config, predecessor, catalog


def build_receipt() -> dict[str, Any]:
    config, predecessor, catalog = load_metadata()
    module_path = _canonical_path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _canonical_path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    rows = list(iter_dimensionality_rows(config, predecessor, catalog))
    _require(len(rows) == 61_740, "dimensionality matrix count changed")
    row_count, row_root = _stream_root(rows)
    disposition_counts = Counter(row["current_disposition"] for row in rows)
    domain_counts = {
        domain: dict(
            sorted(
                Counter(
                    row["current_disposition"] for row in rows if row["domain"] == domain
                ).items()
            )
        )
        for domain in ("SPARC", "XCOP")
    }
    checks = synthetic_contract_checks(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "contract_id": config["contract_id"],
        "status": "PASS_3D_FOUNDATION_METADATA_ONLY_ALL_REAL_OBJECTS_NOT_FULL_3D_READY",
        "decision": "FOUNDATION_READY_SOURCE_ACQUISITION_AND_SOLVERS_STILL_REQUIRED",
        "access_accounting": {
            "scientific_response_files_opened": 0,
            "scientific_response_rows_opened": 0,
            "source_payload_files_opened": 0,
            "source_payload_rows_opened": 0,
            "scores_computed": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "bindings": {
            "config": {
                "path": CONFIG_PATH.as_posix(),
                "sha256": file_sha256(_ROOT / CONFIG_PATH),
                "content_sha256": content_sha256(config),
            },
            "module": {
                "path": _CANONICAL_MODULE_PATH.as_posix(),
                "sha256": file_sha256(module_path),
            },
            "test": {
                "path": _CANONICAL_TEST_PATH.as_posix(),
                "sha256": file_sha256(test_path),
            },
            "predecessors": config["predecessor_bindings"],
        },
        "catalog": {
            "mechanisms": len(catalog),
            "TWELL": 400,
            "GP01": 7,
            "gravity_light_ontology": 13,
            "objects": 147,
            "SPARC": 139,
            "XCOP": 8,
        },
        "dimensionality_matrix": {
            "rows": row_count,
            "stream_sha256": row_root,
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "domain_disposition_counts": domain_counts,
            "full_3d_source_ready_rows": disposition_counts["FULL_3D_SOURCE_READY"],
            "data_eligible_rows": sum(row["data_eligible"] for row in rows),
            "scored_rows": sum(row["scored"] for row in rows),
        },
        "synthetic_contract_checks": {
            "total": len(checks),
            "passed": sum(checks.values()),
            "checks": checks,
            "scope": "fixture-definition invariants only; no promoted 3-D field solver",
        },
        "source_requirements": config["current_object_source_contract"],
        "theory_expansion_queue": config["theory_expansion_queue"],
        "anti_loop_rules": config["anti_loop_rules"],
        "claim_boundary": config["claim_boundary"],
        "section_sha256": config["section_sha256"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(stored: Mapping[str, Any]) -> None:
    observed = stored.get("content_sha256")
    body = {key: value for key, value in stored.items() if key != "content_sha256"}
    _require(observed == content_sha256(body), "receipt self-hash changed")
    expected = build_receipt()
    _require(dict(stored) == expected, "receipt is not a deterministic rebuild")


def _output_path() -> Path:
    _require(OUTPUT_PATH == _CANONICAL_OUTPUT_PATH, "output path changed")
    path = (_ROOT / OUTPUT_PATH).resolve()
    _require(path.is_relative_to(_ROOT), "output escaped repository")
    return path


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


def validate_receipt() -> None:
    path = _output_path()
    stored = _read_json(path, "3-D source/geometry receipt")
    validate_receipt_payload(stored)


def summary() -> dict[str, Any]:
    receipt = build_receipt()
    return {
        "status": receipt["status"],
        "decision": receipt["decision"],
        "catalog": receipt["catalog"],
        "dimensionality_matrix": receipt["dimensionality_matrix"],
        "synthetic_contract_checks": receipt["synthetic_contract_checks"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "summary"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "validate":
        validate_receipt()
        print("VALID")
    else:
        print(json.dumps(summary(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
