"""Freeze five real-source RG fine/coarse radius gates before velocity scoring."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1 as mechanics,
)
from sigma_theory_compiler import (
    open_gravity_rg_things_heracles_s4g_model_lifted_3d_source_builder_v1 as sources,
)

CONFIG_PATH = Path("configs/open_gravity_rg_things_heracles_s4g_scoring_resolution_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_heracles_s4g_scoring_resolution_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_heracles_s4g_scoring_resolution_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-heracles-s4g-scoring-resolution-v1/receipt.json"
)

_CONFIG_RAW_SHA256 = "9c0d94eafa4e7792191ef9962aea45d704ad3e6fce5f61d46096027083e8dcc5"
_CONFIG_CONTENT_SHA256 = "0ff5f8a424cd34525834d2d506b808fd90b4991f43e54eea95c337b46e759dae"
_MODULE_SEMANTIC_SHA256 = "eeb7aab7ad4ab408d7596baeb52f89f3aa6c3f6ac9daeb0f6334a88e57f1c8a1"
_TEST_RAW_SHA256 = "5cfebdc0d2bfc0f977602ca98960172d50d65acbe1a656e9003113611c0d6c0a"
_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f*]{64}("\r?\n)')
_SCHEMA = "invariant-open-gravity-rg-things-heracles-s4g-scoring-resolution-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-things-heracles-s4g-scoring-resolution-receipt-1.0"
_OBJECTS = ("NGC2903", "NGC2976", "NGC3198", "NGC3521", "NGC4214")


class ScoringResolutionError(RuntimeError):
    """Raised when a source, benchmark, or numerical scoring gate changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoringResolutionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: Path | str) -> Path:
    root = _root().resolve()
    candidate = (root / relative).resolve()
    _require(candidate == root or root in candidate.parents, "path escaped repository")
    return candidate


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    clean = dict(value) if type(value) is dict else value
    if type(clean) is dict:
        clean.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(clean)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", path.read_bytes())
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringResolutionError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "SOURCE_ONLY_FIVE_OBJECT_RG_SCORING_RESOLUTION_GATE",
        "status changed",
    )
    _require(config["objects"] == list(_OBJECTS), "objects changed")
    _require(len(config["predecessor_bindings"]) == 3, "predecessors changed")
    source = config["source_cell"]
    _require(
        source["id"]
        == "ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.136986301369863:HG200:CENTRAL_Q0_0P13",
        "source cell changed",
    )
    _require(source["selection_from_response"] is False, "response source selection enabled")
    grid = config["grid_contract"]
    _require(grid["fine_nodes_per_axis"] == 241, "fine grid changed")
    _require(grid["convergence_nodes_per_axis"] == 193, "convergence grid changed")
    _require(grid["fine_spacing_kpc"] == 0.25, "fine spacing changed")
    _require(grid["convergence_spacing_kpc"] == 0.3125, "convergence spacing changed")
    _require(grid["radial_points"] == 291, "radial grid changed")
    operator = config["operator_contract"]
    _require(
        operator["operators"] == ["NEWTON_3D_DST", "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"],
        "operators changed",
    )
    _require(
        operator["published_parameter_id"] == "DISKMASS_UNIVERSAL_MEDIAN", "parameters changed"
    )
    _require(operator["epsilon_0"] == 0.661 and operator["Q"] == 1.79, "published values changed")
    _require(operator["log10_rho_c_g_cm3"] == -24.54, "published density changed")
    _require(operator["response_parameter_fitting"] is False, "response fitting enabled")
    benchmark = config["benchmark_contract"]
    _require(benchmark["radius_gate_is_response_blind"] is True, "radius leakage enabled")
    _require(
        benchmark["fine_vs_convergence_radial_relative_difference_max"] == 0.05, "gate changed"
    )
    boundary = config["scientific_boundary"]
    _require(boundary["unique_source_files_opened_per_build"] == 35, "source count changed")
    _require(
        boundary["response_files_opened"] == boundary["response_rows_opened"] == 0,
        "response access enabled",
    )
    _require(boundary["scores_computed"] == boundary["tuning_calls"] == 0, "scoring enabled")
    claims = config["claim_boundary"]
    _require(claims["paper_and_real_source_anchored"] is True, "source claim lost")
    _require(claims["response_blind_radius_mask_established"] is True, "mask claim lost")
    _require(
        not any(
            claims[key]
            for key in (
                "observational_fit_tested",
                "refracted_gravity_preferred",
                "all_393_source_cells_high_resolution",
                "lensing_closure_established",
                "relativistic_completion_established",
                "novelty_established",
                "publication_ready",
            )
        ),
        "claim ceiling exceeded",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")
    return config


def validate_predecessors(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["predecessor_bindings"]:
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(file_sha256(path) == artifact["sha256"], "predecessor bytes changed")
        receipt_artifact = next(
            row for row in binding["artifacts"] if row["path"].endswith("/receipt.json")
        )
        receipt = _read_json(_repo_path(receipt_artifact["path"]), "predecessor receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            "predecessor receipt content changed",
        )
        receipts[binding["role"]] = receipt
    source = receipts["FIVE_OBJECT_REAL_SOURCE_BUILDER"]
    _require(source["cell_count"] == 393, "source cell count changed")
    _require(all(source["benchmarks"]["passed"].values()), "source benchmark failed")
    mechanics_receipt = receipts["AUDITED_HIGH_RESOLUTION_DST_PCG_MECHANICS"]
    _require(mechanics_receipt["all_object_gates_pass"] is True, "solver predecessor failed")
    primary = receipts["PRIMARY_PAPER_RG_OPERATOR_BENCHMARK"]
    _require(primary["benchmark_suite"]["failed"] == 0, "primary RG benchmark failed")
    return receipts


def _source_evidence(
    config: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, str], Path],
    dict[str, dict[str, Any]],
]:
    source_config = sources.load_config()
    acquisition, geometry, _operator = sources._load_dependencies(source_config)
    paths = sources._source_paths(source_config, acquisition)
    receipt = _read_json(_repo_path(sources.OUTPUT_PATH), "source-builder receipt")
    profile_path = _repo_path(receipt["private_profile_path"])
    _require(file_sha256(profile_path) == receipt["private_profile_raw_sha256"], "profiles changed")
    private = _read_json(profile_path, "source profiles")
    _require(
        private["content_sha256"] == receipt["private_profile_content_sha256"],
        "profile content changed",
    )
    expected = {
        (object_row["object_id"], cell["cell_id"]): cell
        for object_row in private["objects"]
        for cell in object_row["cell_summaries"]
    }
    _require(len(expected) == 393, "source summary ledger changed")
    geometry_by_id = {row["object_id"]: row for row in geometry["objects"]}
    return source_config, acquisition, geometry_by_id, paths, expected


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    predecessors = validate_predecessors(config)
    benchmarks = mechanics.run_target_free_benchmarks(config)
    _require(benchmarks["all_pass"] is True, "target-free solver benchmark failed")
    source_config, _acquisition, geometry, paths, expected = _source_evidence(config)
    bridge_config = mechanics.bridge.load_config()
    source_id = config["source_cell"]["id"]
    object_rows: list[dict[str, Any]] = []
    for object_id in _OBJECTS:
        metadata = sources.geometry_variants(source_config, geometry[object_id])[0]
        images = sources._load_images(object_id, paths)
        maps = sources._maps(source_config, metadata, images)
        _rhalf_pc, exponential_scale_pc = sources._scale_length(maps)
        convergence = mechanics._solve_source_grid(
            config,
            bridge_config,
            maps,
            exponential_scale_pc=exponential_scale_pc,
            expected_source=expected[(object_id, source_id)],
            nodes=int(config["grid_contract"]["convergence_nodes_per_axis"]),
        )
        fine = mechanics._solve_source_grid(
            config,
            bridge_config,
            maps,
            exponential_scale_pc=exponential_scale_pc,
            expected_source=expected[(object_id, source_id)],
            nodes=int(config["grid_contract"]["fine_nodes_per_axis"]),
        )
        object_rows.append(mechanics._adjudicate_object(config, object_id, fine, convergence))
        del maps, images, fine, convergence
        gc.collect()
    all_gates = all(row["all_object_gates_pass"] for row in object_rows)
    eligible = sum(row["eligible_radius_count"] for row in object_rows)
    ineligible = sum(row["ineligible_radius_count"] for row in object_rows)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": (
            "PASS_FIVE_OBJECT_SOURCE_ONLY_RG_SCORING_RESOLUTION_MASK"
            if all_gates
            else "BLOCK_SOURCE_OR_NUMERICAL_GATE_FAILURE_RETAINED"
        ),
        "decision": (
            "READY_FOR_FIXED_HELD_SPARC_RESPONSE_SCORE"
            if all_gates
            else "RESPONSE_SCORE_BLOCKED_RETAIN_FAILURES"
        ),
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "predecessor_receipt_content_sha256": {
            role: row["content_sha256"] for role, row in predecessors.items()
        },
        "real_source_and_paper_anchors": config["real_source_and_paper_anchors"],
        "source_cell": config["source_cell"],
        "grid_contract": config["grid_contract"],
        "operator_contract": config["operator_contract"],
        "target_free_benchmarks": benchmarks,
        "objects": object_rows,
        "all_object_gates_pass": all_gates,
        "response_blind_radius_summary": {
            "registered_points": len(_OBJECTS) * int(config["grid_contract"]["radial_points"]),
            "eligible_points": eligible,
            "ineligible_points": ineligible,
            "selection_used_velocity_values": False,
        },
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    _require(dict(payload) == build_receipt(config), "receipt differs from exact rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent output differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt() -> str:
    return _atomic_no_clobber(
        _repo_path(OUTPUT_PATH), canonical_bytes(build_receipt(load_config()))
    )


def check_receipt() -> str:
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    _require(path.read_bytes() == canonical_bytes(build_receipt(load_config())), "receipt changed")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        print(check_receipt())
    else:
        config = load_config()
        print(
            json.dumps(
                {
                    "status": config["status"],
                    "output_exists": _repo_path(OUTPUT_PATH).exists(),
                    "response_rows_opened": 0,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
