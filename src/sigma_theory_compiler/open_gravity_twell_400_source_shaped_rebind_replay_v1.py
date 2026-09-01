"""Append-only, response-blind TWELL-400 source-shaped rebind and replay.

The successor binds every final-v3 card and parameter cell to five independently
audited source-shaped releases.  It executes only the static radial formulas that
the X-COP spherical-lift packet can honestly supply.  No response-valued NPZ member
is loaded and no scientific score is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_static_radial_adapter_v1 as static_adapter

CONFIG_PATH = Path("configs/open_gravity_twell_400_source_shaped_rebind_replay_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_twell_400_source_shaped_rebind_replay_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_twell_400_source_shaped_rebind_replay_v1.py")
OUTPUT_DIR = Path("runs/gravity/open-gravity-twell-400-source-shaped-rebind-replay-v1")
COMPATIBILITY_PATH = OUTPUT_DIR / "compatibility-ledger.jsonl"
DISPOSITION_PATH = OUTPUT_DIR / "parameter-cell-disposition-ledger.jsonl"
BINDINGS_PATH = OUTPUT_DIR / "execution-bindings.jsonl"
SOURCE_ROWS_PATH = OUTPUT_DIR / "source-projections.jsonl"
SOURCE_VALUES_PATH = OUTPUT_DIR / "source-projections.npz"
UNIQUE_EXECUTIONS_PATH = OUTPUT_DIR / "unique-executions.jsonl"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.npz"
REPLAY_PATH = OUTPUT_DIR / "replay-ledger.jsonl"
EQUIVALENCE_PATH = OUTPUT_DIR / "equivalence-ties.json"
INVARIANCE_PATH = OUTPUT_DIR / "invariance-gates.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"

CONFIG_SCHEMA = "open-gravity-twell-400-source-shaped-rebind-replay-config-1.0"
RECEIPT_SCHEMA = "open-gravity-twell-400-source-shaped-rebind-replay-receipt-1.0"
DECISION = "SOURCE_SHAPED_REBIND_REPLAY_FROZEN_AWAITING_DISTINCT_INDEPENDENT_AUDIT"
DERIVATION_PROGRAM = {
    "id": "CARTESIAN_RADIUS_SHELL_MEAN_TOTAL_BARYON_PROFILE_TO_257_POINT_STATIC_RADIAL_V1",
    "input": "spherical-lifted 17^3 total baryonic mass density on centered Cartesian SI grid",
    "radial_reduction": "mean density at every exact integer Cartesian radius-squared shell",
    "mass_rule": "trapezoidal 4*pi*rho*r^2 integration with regular constant-density origin",
    "grid_rule": "linear enclosed-mass interpolation to xi=linspace(0,1,257)",
    "drivers": ["D01_ACC", "D02_POT", "D03_RAD", "D04_RHO", "D05_SIG", "D06_SLOPE", "D07_TIDE"],
    "forbidden_inference": "D13_GASF is not derivable from total baryonic density",
}
DERIVATION_PROGRAM_SHA256 = ""

_TOKEN = re.compile(r"[^a-z0-9]+")
_G_SI = 6.67430e-11
_KPC_M = 3.085677581491367e19
_STATIC_ARCHITECTURES = frozenset(static_adapter.STATIC_ARCHITECTURES)
_TEMPORAL_ARCHITECTURES = frozenset(static_adapter.TIME_SOURCE_BLOCKS)
_SUPPORTED_DRIVERS = frozenset(static_adapter.XCOP_DRIVERS)
_SUPPORTED_COMPOUNDS = frozenset(static_adapter.COMPOUND_IDS)
_D13 = "D13_GASF"
_COMPOUND_ARCHITECTURES = {
    "X01": "A02_CLOCK",
    "X05": "A01_LAPSE",
    "X10": "A11_DERIV_SCREEN",
    "X13": "A08_PERMITTIVITY",
    "X17": "A12_MASSIVE",
    "X18": "A13_MIXED_MODE",
}


class TwellSourceRebindError(RuntimeError):
    """Raised when a frozen source-only rebind invariant fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TwellSourceRebindError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TwellSourceRebindError(f"noncanonical value: {error}") from error


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


DERIVATION_PROGRAM_SHA256 = content_sha256(DERIVATION_PROGRAM)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    """Hash an array with the frozen source-packet array convention."""

    array = np.ascontiguousarray(np.asarray(value))
    _require(array.dtype.name in {"float64", "int64"}, "unsupported array dtype")
    _require(bool(np.all(np.isfinite(array))), "nonfinite array")
    digest = hashlib.sha256()
    digest.update(array.dtype.name.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _token(value: str) -> str:
    return _TOKEN.sub("_", value.lower()).strip("_")


def _repo_path(root: Path, value: str) -> Path:
    parsed = PurePosixPath(value.replace("\\", "/"))
    _require(
        not parsed.is_absolute() and all(part not in {"", ".", ".."} for part in parsed.parts),
        f"path escaped repository: {value}",
    )
    repo = root.resolve(strict=True)
    result = (repo / parsed.as_posix()).resolve(strict=False)
    _require(result == repo or repo in result.parents, f"path escaped repository: {value}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TwellSourceRebindError(f"could not load JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sealed_row(row: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(row)
    _require(field not in result, f"seal field already present: {field}")
    result[field] = content_sha256(result)
    return result


def _receipt_content_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("content_sha256", None)
    return content_sha256(payload)


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    config = _read_json(_repo_path(root, CONFIG_PATH.as_posix()))
    validate_config(root.resolve(), config)
    return config


def _validate_source_release(root: Path, row: Mapping[str, Any]) -> None:
    receipt_path = _repo_path(root, str(row["receipt_path"]))
    audit_path = _repo_path(root, str(row["audit_path"]))
    scenario_path = _repo_path(root, str(row["scenario_path"]))
    _require(file_sha256(receipt_path) == row["receipt_raw_sha256"], "source receipt drift")
    _require(file_sha256(audit_path) == row["audit_raw_sha256"], "source audit drift")
    _require(file_sha256(scenario_path) == row["scenario_raw_sha256"], "scenario stream drift")
    receipt = _read_json(receipt_path)
    audit = _read_json(audit_path)
    _require(receipt.get("content_sha256") == row["receipt_content_sha256"], "source receipt seal drift")
    _require(audit.get("content_sha256") == row["audit_content_sha256"], "source audit seal drift")
    if "values_path" in row:
        _require(
            file_sha256(_repo_path(root, str(row["values_path"]))) == row["values_raw_sha256"],
            "source values archive drift",
        )


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema", "package_id", "version", "status", "claim_class", "output_directory",
        "frozen_predecessors", "twell_contract", "source_releases", "feature_contracts",
        "rebind_contract", "exclusions",
    }
    _require(set(config) == expected_keys, "config keys changed")
    _require(config["schema"] == CONFIG_SCHEMA, "config schema changed")
    _require(config["package_id"] == OUTPUT_DIR.name, "package ID changed")
    _require(config["version"] == "v1.0.0", "package version changed")
    _require(
        config["status"] == "PRE_RESPONSE_SOURCE_ONLY_REBIND_REPLAY",
        "package status changed",
    )
    _require(config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL", "claim class changed")
    _require(config["output_directory"] == OUTPUT_DIR.as_posix(), "output path changed")
    predecessors = list(config["frozen_predecessors"])
    _require(len(predecessors) == 14, "predecessor inventory changed")
    _require(len({row["id"] for row in predecessors}) == 14, "duplicate predecessor")
    for row in predecessors:
        path = _repo_path(root, str(row["path"]))
        _require(path.is_file(), f"missing predecessor: {row['id']}")
        _require(file_sha256(path) == row["raw_sha256"], f"predecessor drift: {row['id']}")
        if "content_sha256" in row:
            payload = _read_json(path)
            content = payload.get("content_sha256", payload.get("receipt_content_sha256"))
            _require(content == row["content_sha256"], f"predecessor seal drift: {row['id']}")
    twell = config["twell_contract"]
    _require(
        (twell["card_count"], twell["atomic_card_count"], twell["compound_card_count"])
        == (400, 380, 20),
        "TWELL card counts changed",
    )
    _require(
        (
            twell["parameter_cell_count"], twell["provisional_static_card_count"],
            twell["temporal_card_count"], twell["missing_driver_card_count"],
        ) == (1184, 126, 84, 190),
        "TWELL gap counts changed",
    )
    _require(set(twell["supported_architectures"]) == _STATIC_ARCHITECTURES, "architecture set drift")
    _require(set(twell["temporal_architectures"]) == _TEMPORAL_ARCHITECTURES, "temporal set drift")
    _require(set(twell["supported_drivers"]) == _SUPPORTED_DRIVERS, "driver set drift")
    _require(set(twell["supported_compounds"]) == _SUPPORTED_COMPOUNDS, "compound set drift")
    releases = list(config["source_releases"])
    _require(len(releases) == 5, "source release inventory changed")
    _require(len({row["release_id"] for row in releases}) == 5, "duplicate source release")
    _require(sum(int(row["scenario_count"]) for row in releases) == 1179, "scenario accounting drift")
    for row in releases:
        _require(row["feature_contract"] in config["feature_contracts"], "unknown feature contract")
        _validate_source_release(root, row)
    full3d = config["feature_contracts"]["FULL3D_NINE_FEATURES"]
    _require(len(full3d) == 9, "full-3D feature count changed")
    _require(
        next(row for row in full3d if row["id"] == "source.scalar.mass-density")["unit"]
        == "kg m^-3",
        "density unit changed",
    )
    void = config["feature_contracts"]["VOID_FOURTEEN_FEATURES"]
    _require(len(void) == 14, "void feature count changed")
    for feature_id in ("source.scalar.delta-h-km-s-mpc", "source.scalar.h-m-km-s-mpc"):
        _require(
            next(row for row in void if row["id"] == feature_id)["unit"]
            == "km s^-1 Mpc^-1",
            "void Hubble-rate unit changed",
        )
    rebind = config["rebind_contract"]
    _require(
        (
            rebind["xcop_executable_card_count"], rebind["xcop_executable_parameter_cell_count"],
            rebind["xcop_d13_blocked_card_count"],
            rebind["xcop_d13_blocked_parameter_cell_count"], rebind["compatibility_row_count"],
            rebind["replay_row_count"], rebind["unique_execution_count"],
        ) == (110, 324, 16, 46, 2000, 62208, 2592),
        "rebind counts changed",
    )
    _require(rebind["xcop_derivation"] == DERIVATION_PROGRAM["id"], "derivation ID changed")
    _require(rebind["deterministic_execution_repetitions"] == 2, "replay repetitions changed")
    exclusions = config["exclusions"]
    _require(all(value is False for value in exclusions.values()), "an exclusion was weakened")


def _card_stream_binding(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(row for row in config["frozen_predecessors"] if row["id"] == "TWELL_400_FINAL_V3_CARDS")


def load_cards(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    binding = _card_stream_binding(config)
    path = _repo_path(root, str(binding["path"]))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(len(rows) == 400, "card stream count changed")
    expected_root = content_sha256([content_sha256(row) for row in rows])
    _require(
        expected_root == config["twell_contract"]["ordered_line_root_sha256"],
        "ordered card stream root changed",
    )
    seen_concepts: set[str] = set()
    seen_cells: set[str] = set()
    for index, row in enumerate(rows):
        card = row["card"]
        concept_id = str(row["concept_id"])
        _require(row["order_index"] == index, "card order index changed")
        _require(concept_id == card["stable_concept_id"], "concept identity changed")
        _require(concept_id not in seen_concepts, "duplicate concept ID")
        seen_concepts.add(concept_id)
        _require(content_sha256(card) == row["card_sha256"], "card SHA mismatch")
        _require(row["manifest_input"]["card_sha256"] == row["card_sha256"], "manifest card mismatch")
        formula_sha = card["hashes"]["formula_sha256"]
        _require(row["manifest_input"]["formula_sha256"] == formula_sha, "formula SHA mismatch")
        cells = card["parameter_cells"]
        _require(len(cells) == row["parameter_cell_count"], "parameter-cell count mismatch")
        for cell in cells:
            _require(cell["cell_id"] not in seen_cells, "duplicate parameter-cell ID")
            seen_cells.add(cell["cell_id"])
            _require(cell["frozen"] is True, "unfrozen parameter cell")
    _require(len(seen_cells) == 1184, "parameter-cell coverage changed")
    return rows


def classify_card(row: Mapping[str, Any]) -> str:
    architecture = str(row["architecture_id"])
    concept_id = str(row["concept_id"])
    drivers = set(row["driver_ids"])
    if architecture in _TEMPORAL_ARCHITECTURES:
        return "TEMPORAL_ARCHITECTURE"
    if row["entry_kind"] == "ATOMIC" and architecture in _STATIC_ARCHITECTURES and drivers <= _SUPPORTED_DRIVERS:
        return "PROVISIONAL_STATIC"
    if concept_id in _SUPPORTED_COMPOUNDS:
        return "PROVISIONAL_STATIC"
    return "MISSING_DRIVER_OR_COMPOUND_ADAPTER"


def _class_counts(cards: Sequence[Mapping[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    card_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    for row in cards:
        classification = classify_card(row)
        card_counts[classification] += 1
        cell_counts[classification] += int(row["parameter_cell_count"])
    return card_counts, cell_counts


def _feature_contract_sha(config: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    return content_sha256(config["feature_contracts"][source["feature_contract"]])


def build_compatibility_ledger(
    config: Mapping[str, Any], cards: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in cards:
        classification = classify_card(card)
        for source in config["source_releases"]:
            if source["release_id"] != "XCOP_REAL_SOURCE_SHAPED_SYNTHETIC_V1":
                status = "INCOMPATIBLE_FEATURE_SET"
                reason = "STATIC_SPHERICAL_RADIAL_ABI_NOT_SUPPORTED_BY_SOURCE_GEOMETRY"
            elif classification == "TEMPORAL_ARCHITECTURE":
                status = "SOURCE_BLOCKED"
                reason = "TYPED_TIME_HISTORY_ABSENT_GW_TIMESERIES_V3_EXCLUDED_PENDING_AUDIT"
            elif classification == "MISSING_DRIVER_OR_COMPOUND_ADAPTER":
                status = "SOURCE_BLOCKED"
                reason = "DECLARED_DRIVER_OR_COMPOUND_ADAPTER_ABSENT_FROM_SOURCE_PACKET"
            elif _D13 in card["driver_ids"]:
                status = "SOURCE_BLOCKED"
                reason = "D13_GAS_FRACTION_ABSENT_TOTAL_BARYON_DENSITY_CANNOT_SPLIT_COMPONENTS"
            else:
                status = "EXECUTABLE"
                reason = "AUDITED_STATIC_SPHERICAL_TOTAL_BARYON_SOURCE_PROJECTION_AVAILABLE"
            feature_contract = config["feature_contracts"][source["feature_contract"]]
            base = {
                "formula_id": card["concept_id"],
                "domain": source["domain"],
                "source_release": source["release_id"],
                "status": status,
                "reason": reason,
                "classification": classification,
                "architecture_id": card["architecture_id"],
                "driver_ids": card["driver_ids"],
                "card_sha256": card["card_sha256"],
                "formula_sha256": card["card"]["hashes"]["formula_sha256"],
                "source_release_sha256": source["receipt_raw_sha256"],
                "source_receipt_content_sha256": source["receipt_content_sha256"],
                "source_audit_sha256": source["audit_raw_sha256"],
                "source_feature_contract": source["feature_contract"],
                "source_feature_contract_sha256": _feature_contract_sha(config, source),
                "source_feature_ids": [item["id"] for item in feature_contract],
                "geometry": source["geometry"],
                "time": source["time"],
                "coordinate_frame": source["frame"],
            }
            rows.append(_sealed_row(base, "compatibility_sha256"))
    _require(len(rows) == 2000, "compatibility row coverage changed")
    counts = Counter(row["status"] for row in rows)
    _require(counts == {"EXECUTABLE": 110, "SOURCE_BLOCKED": 290, "INCOMPATIBLE_FEATURE_SET": 1600}, "compatibility counts changed")
    return rows


def build_disposition_ledger(cards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in cards:
        classification = classify_card(card)
        if classification == "TEMPORAL_ARCHITECTURE":
            status = "SOURCE_BLOCKED"
            reason = "TEMPORAL_HISTORY_INITIAL_STATE_AND_CADENCE_ABSENT"
        elif classification == "MISSING_DRIVER_OR_COMPOUND_ADAPTER":
            status = "SOURCE_BLOCKED"
            reason = "MISSING_DECLARED_SOURCE_DRIVER_OR_COMPOUND_ADAPTER"
        elif _D13 in card["driver_ids"]:
            status = "SOURCE_BLOCKED"
            reason = "D13_GAS_FRACTION_NOT_PRESENT_IN_XCOP_FORMULA_FEATURE_PACKET"
        else:
            status = "COMPLETED_WITH_SCENARIO_LEVEL_NUMERICAL_VALIDITY_RETAINED"
            reason = "ADMITTED_TO_COMMON_ABI_ON_EVERY_XCOP_SCENARIO"
        for cell in card["card"]["parameter_cells"]:
            base = {
                "cell_id": cell["cell_id"],
                "formula_id": card["concept_id"],
                "status": status,
                "reason": reason,
                "classification": classification,
                "architecture_id": card["architecture_id"],
                "driver_ids": card["driver_ids"],
                "parameters": cell["value"],
                "parameter_unit": cell["unit"],
                "parameter_kind": cell["parameter"],
                "parameter_cell_sha256": content_sha256(cell),
                "card_sha256": card["card_sha256"],
                "formula_sha256": card["card"]["hashes"]["formula_sha256"],
                "equivalence_family_id": card["equivalence_family_id"],
            }
            rows.append(_sealed_row(base, "disposition_sha256"))
    _require(len(rows) == 1184, "disposition coverage changed")
    counts = Counter(row["status"] for row in rows)
    _require(
        counts
        == {
            "COMPLETED_WITH_SCENARIO_LEVEL_NUMERICAL_VALIDITY_RETAINED": 324,
            "SOURCE_BLOCKED": 860,
        },
        "disposition status counts changed",
    )
    return rows


def build_execution_bindings(
    config: Mapping[str, Any], cards: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    source = next(
        row for row in config["source_releases"]
        if row["release_id"] == "XCOP_REAL_SOURCE_SHAPED_SYNTHETIC_V1"
    )
    feature_contract = config["feature_contracts"][source["feature_contract"]]
    rows: list[dict[str, Any]] = []
    for card in cards:
        if classify_card(card) != "PROVISIONAL_STATIC":
            continue
        executable = _D13 not in card["driver_ids"]
        base = {
            "formula_id": card["concept_id"],
            "status": "EXECUTABLE" if executable else "SOURCE_BLOCKED",
            "block_reason": None if executable else "D13_GAS_FRACTION_ABSENT",
            "callable": config["rebind_contract"]["common_abi"] if executable else None,
            "architecture_id": card["architecture_id"],
            "driver_ids": card["driver_ids"],
            "parameter_cell_ids": [cell["cell_id"] for cell in card["card"]["parameter_cells"]],
            "parameter_schema": [
                {"cell_id": cell["cell_id"], "unit": cell["unit"], "value": cell["value"]}
                for cell in card["card"]["parameter_cells"]
            ],
            "required_source_features": feature_contract,
            "source_derivation_program_sha256": DERIVATION_PROGRAM_SHA256,
            "geometry_support": ["spherical-lifted3d", "static-radial-source-projection"],
            "time_support": ["static"],
            "coordinate_frame": "solver-source-centered-radial",
            "emitted_features": config["rebind_contract"]["emitted_features"],
            "limits": card["card"]["limiting_cases"],
            "boundaries": card["card"]["boundaries"],
            "initial_conditions": card["card"]["initial_conditions"],
            "resource_ceiling": config["rebind_contract"]["resource_ceiling"],
            "card_sha256": card["card_sha256"],
            "formula_sha256": card["card"]["hashes"]["formula_sha256"],
            "static_adapter_module_sha256": "7db918df47f612df3c42792c25d407bf5c485db6f7830f8bd690b57b95b5968f",
            "static_adapter_receipt_sha256": "8238b9fe605d14f3fe637c82068a17463634579d3cf396c7ce09bb2642ebb931",
            "source_release_sha256": source["receipt_raw_sha256"],
            "source_scenarios_sha256": source["scenario_raw_sha256"],
            "source_values_sha256": source["values_raw_sha256"],
        }
        rows.append(_sealed_row(base, "adapter_binding_sha256"))
    _require(len(rows) == 126, "execution binding count changed")
    _require(Counter(row["status"] for row in rows) == {"EXECUTABLE": 110, "SOURCE_BLOCKED": 16}, "execution binding statuses changed")
    return rows


def _source_key(object_id: str, feature_id: str) -> str:
    return f"source__{_token(object_id)}__{_token(feature_id)}"


def load_xcop_scenario_metadata(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    source = next(
        row for row in config["source_releases"]
        if row["release_id"] == "XCOP_REAL_SOURCE_SHAPED_SYNTHETIC_V1"
    )
    path = _repo_path(root, source["scenario_path"])
    scenarios: list[dict[str, Any]] = []
    source_refs: dict[str, dict[str, Any]] = {}
    expected_contract = config["feature_contracts"][source["feature_contract"]]
    expected_ids = [row["id"] for row in expected_contract]
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        descriptor = raw["scenario"]
        _require(content_sha256(descriptor) == raw["scenario_sha256"], "scenario SHA mismatch")
        _require(
            (descriptor["domain"], descriptor["geometry_mode"], descriptor["time_mode"], descriptor["coordinate_frame"])
            == (source["domain"], source["geometry"], source["time"], source["frame"]),
            "X-COP scenario capability changed",
        )
        refs = descriptor["formula_features"]
        _require([row["element_id"] for row in refs] == expected_ids, "X-COP feature order changed")
        for ref, contract in zip(refs, expected_contract, strict=True):
            _require(ref["unit"] == contract["unit"] and ref["axes"] == contract["axes"], "X-COP feature typing changed")
            _require(ref["frame"] == source["frame"], "X-COP feature frame changed")
        object_id = _token(raw["object_id"])
        ref_map = {row["element_id"]: row for row in refs}
        if object_id in source_refs:
            _require(source_refs[object_id] == ref_map, "source refs changed across nuisance/truth worlds")
        else:
            source_refs[object_id] = ref_map
        lineage = descriptor["seed_lineage"]
        scenarios.append(
            {
                "scenario_id": lineage["scenario_id"],
                "scenario_sha256": raw["scenario_sha256"],
                "object_id": object_id,
                "truth_formula_id": raw["truth_formula_id"],
                "truth_world_id": raw["truth_world_id"],
                "noise_family": raw["noise"]["family"],
                "derived_seed": str(raw["noise"]["derived_seed"]),
                "suite_seed": lineage["suite_seed"],
                "nuisance_draw": lineage["nuisance_draw"],
                "operator_draw": lineage["operator_draw"],
            }
        )
    _require(len(scenarios) == 192 and len(source_refs) == 8, "X-COP scenario/object count changed")
    _require(len({row["scenario_sha256"] for row in scenarios}) == 192, "duplicate scenario")
    _require(Counter(row["object_id"] for row in scenarios) == {key: 24 for key in source_refs}, "per-object scenario count changed")
    return scenarios, source_refs


def _log_slope(radius: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    positive = (radius > 0.0) & (acceleration > 0.0)
    indices = np.flatnonzero(positive)
    _require(indices.size >= 3, "radial acceleration has too few positive nodes")
    slope_positive = -np.gradient(
        np.log(acceleration[indices]), np.log(radius[indices]), edge_order=2
    )
    result = np.empty_like(acceleration)
    result[indices] = slope_positive
    result[: indices[0]] = slope_positive[0]
    return result


def _project_total_baryon_density(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, density: np.ndarray
) -> dict[str, Any]:
    coordinates = [np.asarray(value, dtype=np.float64) for value in (x, y, z)]
    _require(all(value.shape == (17,) for value in coordinates), "coordinate shape changed")
    centered = [value - 0.5 * (value[0] + value[-1]) for value in coordinates]
    spacing = [np.diff(value) for value in centered]
    _require(
        all(np.allclose(value, value[0], rtol=2.0e-15, atol=0.0) for value in spacing),
        "grid is not uniform",
    )
    _require(np.allclose([value[0] for value in spacing], spacing[0][0], rtol=1.0e-14, atol=0.0), "grid spacing differs by axis")
    _require(density.shape == (17, 17, 17), "density shape changed")
    _require(bool(np.all(np.isfinite(density))) and bool(np.all(density >= 0.0)), "density invalid")
    dx = float(spacing[0][0])
    indices = np.arange(17, dtype=np.int64) - 8
    radius_squared_index = (
        indices[:, None, None] ** 2
        + indices[None, :, None] ** 2
        + indices[None, None, :] ** 2
    )
    source_radius: list[float] = []
    source_density: list[float] = []
    shell_multiplicity: list[int] = []
    for shell in sorted(int(value) for value in np.unique(radius_squared_index) if value > 0):
        values = np.asarray(density[radius_squared_index == shell], dtype=np.float64)
        mean = float(np.mean(values))
        if mean > 0.0:
            source_radius.append(math.sqrt(shell) * dx)
            source_density.append(mean)
            shell_multiplicity.append(int(values.size))
    radius = np.asarray(source_radius, dtype=np.float64)
    rho = np.asarray(source_density, dtype=np.float64)
    _require(radius.size == 34 and bool(np.all(np.diff(radius) > 0.0)), "radial shell support changed")
    _require(bool(np.all(rho > 0.0)), "positive shell density support changed")
    mass = np.empty_like(radius)
    mass[0] = 4.0 * math.pi * rho[0] * radius[0] ** 3 / 3.0
    integrand = 4.0 * math.pi * rho * radius * radius
    mass[1:] = mass[0] + np.cumsum(
        0.5 * (integrand[:-1] + integrand[1:]) * np.diff(radius)
    )
    xi = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    grid_radius = xi * radius[-1]
    baryonic_mass = np.interp(grid_radius, radius, mass)
    inner = grid_radius < radius[0]
    baryonic_mass[inner] = mass[0] * (grid_radius[inner] / radius[0]) ** 3
    baryonic_mass[0] = 0.0
    g_b = np.zeros_like(grid_radius)
    g_b[1:] = _G_SI * baryonic_mass[1:] / grid_radius[1:] ** 2
    derivative = np.gradient(baryonic_mass, grid_radius, edge_order=2)
    rho_b = np.empty_like(baryonic_mass)
    rho_b[1:] = np.maximum(
        derivative[1:] / (4.0 * math.pi * grid_radius[1:] ** 2), 0.0
    )
    rho_b[0] = 3.0 * baryonic_mass[1] / (4.0 * math.pi * grid_radius[1] ** 3)
    potential = np.zeros_like(grid_radius)
    for index in range(grid_radius.size - 2, -1, -1):
        potential[index] = potential[index + 1] + 0.5 * (
            g_b[index] + g_b[index + 1]
        ) * (grid_radius[index + 1] - grid_radius[index])
    surface_density = np.zeros_like(grid_radius)
    surface_density[1:] = baryonic_mass[1:] / (math.pi * grid_radius[1:] ** 2)
    slope = _log_slope(grid_radius, g_b)
    g_over_r = np.empty_like(g_b)
    g_over_r[1:] = g_b[1:] / grid_radius[1:]
    g_over_r[0] = g_over_r[1]
    tide = math.sqrt(2.0 / 3.0) * np.abs(4.0 * math.pi * _G_SI * rho_b - 3.0 * g_over_r)
    physical = {
        "D01_ACC": g_b,
        "D02_POT": potential,
        "D03_RAD": grid_radius,
        "D04_RHO": rho_b,
        "D05_SIG": surface_density,
        "D06_SLOPE": slope,
        "D07_TIDE": tide,
    }
    normalized = {
        name: static_adapter.normalize_driver(name, value) for name, value in physical.items()
    }
    return {
        "xi": xi,
        "radius_m": grid_radius,
        "physical": physical,
        "normalized": normalized,
        "source_shell_radius_m": radius,
        "source_shell_density_kg_m3": rho,
        "source_shell_multiplicity": np.asarray(shell_multiplicity, dtype=np.int64),
        "baryonic_enclosed_mass_kg": baryonic_mass,
        "metadata": {
            "source_shell_count": int(radius.size),
            "outer_radius_m_hex": float(radius[-1]).hex(),
            "translation_rule": "center each axis by its endpoint midpoint",
            "rotation_rule": "Cartesian-radius shell mean",
            "gas_fraction_inferred": False,
        },
    }


def load_xcop_source_projections(
    root: Path, config: Mapping[str, Any], source_refs: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    source = next(
        row for row in config["source_releases"]
        if row["release_id"] == "XCOP_REAL_SOURCE_SHAPED_SYNTHETIC_V1"
    )
    contract = config["feature_contracts"][source["feature_contract"]]
    values_path = _repo_path(root, source["values_path"])
    projections: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    source_bytes = 0
    with np.load(values_path, allow_pickle=False) as archive:
        for object_id in sorted(source_refs):
            arrays: dict[str, np.ndarray] = {}
            feature_hashes: dict[str, str] = {}
            for feature in contract:
                feature_id = feature["id"]
                key = _source_key(object_id, feature_id)
                _require(key.startswith("source__"), "non-source key requested")
                _require(key in archive.files, f"missing source member: {key}")
                value = np.array(archive[key], copy=True)
                source_bytes += int(value.nbytes)
                digest = array_sha256(value)
                _require(digest == source_refs[object_id][feature_id]["value_sha256"], "source feature value drift")
                _require(value.dtype.name == source_refs[object_id][feature_id]["dtype"], "source dtype drift")
                _require(list(value.shape) == source_refs[object_id][feature_id]["shape"], "source shape drift")
                arrays[feature_id] = value
                feature_hashes[feature_id] = digest
            projection = _project_total_baryon_density(
                arrays["geometry.scalar.x-coordinate"],
                arrays["geometry.scalar.y-coordinate"],
                arrays["geometry.scalar.z-coordinate"],
                arrays["source.scalar.mass-density"],
            )
            projections[object_id] = projection
            projected_hashes = {
                "xi": array_sha256(projection["xi"]),
                "radius_m": array_sha256(projection["radius_m"]),
                "baryonic_enclosed_mass_kg": array_sha256(projection["baryonic_enclosed_mass_kg"]),
                **{
                    f"physical.{name}": array_sha256(value)
                    for name, value in projection["physical"].items()
                },
                **{
                    f"normalized.{name}": array_sha256(value)
                    for name, value in projection["normalized"].items()
                },
            }
            base = {
                "object_id": object_id,
                "source_release": source["release_id"],
                "source_release_sha256": source["receipt_raw_sha256"],
                "source_values_sha256": source["values_raw_sha256"],
                "source_feature_hashes": feature_hashes,
                "source_feature_root_sha256": content_sha256(feature_hashes),
                "derivation_program_sha256": DERIVATION_PROGRAM_SHA256,
                "projected_array_hashes": projected_hashes,
                "projection_metadata": projection["metadata"],
                "geometry": "spherical-lifted3d-to-centered-static-radial",
                "time": "static",
                "coordinate_frame": "solver-source-centered-radial",
                "driver_units": {
                    "D01_ACC": "m s^-2", "D02_POT": "m2 s^-2", "D03_RAD": "m",
                    "D04_RHO": "kg m^-3", "D05_SIG": "kg m^-2", "D06_SLOPE": "1",
                    "D07_TIDE": "s^-2",
                },
                "driver_axes": {name: ["radial"] for name in projection["physical"]},
                "D13_GASF_available": False,
            }
            rows.append(_sealed_row(base, "source_projection_sha256"))
    _require(len(projections) == len(rows) == 8, "source projection count changed")
    return projections, rows, source_bytes


def _projection_arrays(projections: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for object_id, projection in sorted(projections.items()):
        prefix = f"source_projection__{object_id}__"
        result[prefix + "xi"] = projection["xi"]
        result[prefix + "radius_m"] = projection["radius_m"]
        result[prefix + "baryonic_enclosed_mass_kg"] = projection["baryonic_enclosed_mass_kg"]
        result[prefix + "source_shell_radius_m"] = projection["source_shell_radius_m"]
        result[prefix + "source_shell_density_kg_m3"] = projection["source_shell_density_kg_m3"]
        result[prefix + "source_shell_multiplicity"] = projection["source_shell_multiplicity"]
        for name, value in sorted(projection["physical"].items()):
            result[prefix + "physical__" + name.lower()] = value
        for name, value in sorted(projection["normalized"].items()):
            result[prefix + "normalized__" + name.lower()] = value
    return result


def _driver_for_card(card: Mapping[str, Any], source_bundle: Mapping[str, Any]) -> np.ndarray:
    drivers = list(card["driver_ids"])
    _require(_D13 not in drivers, "D13 source block bypassed")
    normalized = source_bundle["normalized"]
    if card["entry_kind"] == "ATOMIC":
        _require(len(drivers) == 1 and drivers[0] in normalized, "atomic driver unavailable")
        return np.asarray(normalized[drivers[0]], dtype=np.float64)
    _require(card["concept_id"] in _SUPPORTED_COMPOUNDS, "compound adapter unavailable")
    return static_adapter.combine_compound_drivers(
        "XCOP_SPHERICAL",
        str(card["concept_id"]),
        {name: normalized[name] for name in drivers},
    )


def execute_twell_static_common_abi(
    card: Mapping[str, Any], source_bundle: Mapping[str, Any], parameter_cell: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute one final-v3 cell on one static source with explicit validity gates."""

    _require(classify_card(card) == "PROVISIONAL_STATIC", "card is not statically adapted")
    _require(_D13 not in card["driver_ids"], "D13 card is source-blocked")
    _require(parameter_cell in card["card"]["parameter_cells"], "parameter cell not in card")
    architecture = str(card["architecture_id"])
    if card["entry_kind"] == "COMPOUND":
        _require(_COMPOUND_ARCHITECTURES[card["concept_id"]] == architecture, "compound architecture changed")
    xi = np.asarray(source_bundle["xi"], dtype=np.float64)
    g_b = np.asarray(source_bundle["physical"]["D01_ACC"], dtype=np.float64)
    driver = _driver_for_card(card, source_bundle)
    primary = static_adapter.apply_static_architecture(
        architecture, driver, g_b, xi, parameter_cell["value"]
    )
    xi_convergence = np.linspace(0.0, 1.0, 129, dtype=np.float64)
    convergence = static_adapter.apply_static_architecture(
        architecture,
        np.interp(xi_convergence, xi, driver),
        np.interp(xi_convergence, xi, g_b),
        xi_convergence,
        parameter_cell["value"],
    )
    convergence_max_abs = float(
        np.max(np.abs(np.asarray(primary["factor"])[::2] - np.asarray(convergence["factor"])))
    )
    diagnostics = {
        "finite": primary["diagnostics"]["finite"] and convergence["diagnostics"]["finite"],
        "positive_factor": primary["diagnostics"]["positive_factor"] and convergence["diagnostics"]["positive_factor"],
        "operator_residual_max_abs_hex": max(
            float(primary["diagnostics"]["operator_residual_max_abs"]),
            float(convergence["diagnostics"]["operator_residual_max_abs"]),
        ).hex(),
        "boundary_residual_max_abs_hex": max(
            float(primary["diagnostics"]["boundary_residual_max_abs"]),
            float(convergence["diagnostics"]["boundary_residual_max_abs"]),
        ).hex(),
        "primary_vs_convergence_max_abs_hex": convergence_max_abs.hex(),
        "primary_vs_convergence_tolerance_hex": float(static_adapter.CONVERGENCE_MAX_ABS_TOLERANCE).hex(),
    }
    numerical_valid = bool(
        diagnostics["finite"]
        and diagnostics["positive_factor"]
        and float.fromhex(diagnostics["operator_residual_max_abs_hex"])
        <= static_adapter.OPERATOR_RESIDUAL_TOLERANCE
        and float.fromhex(diagnostics["boundary_residual_max_abs_hex"])
        <= static_adapter.BOUNDARY_RESIDUAL_TOLERANCE
        and convergence_max_abs <= static_adapter.CONVERGENCE_MAX_ABS_TOLERANCE
    )
    factor = np.asarray(primary["factor"], dtype=np.float64)
    effective = np.asarray(primary["g_eff_m_s2"], dtype=np.float64)
    _require(factor.shape == effective.shape == (257,), "common ABI output shape changed")
    _require(factor.nbytes + effective.nbytes <= 8192, "common ABI output ceiling exceeded")
    health = {
        "kind": "SOURCE_ONLY_NUMERICAL_HEALTH_NOT_SCIENTIFIC_SCORE",
        "factor_min_hex": float(np.min(factor)).hex(),
        "factor_max_hex": float(np.max(factor)).hex(),
        "g_eff_rms_m_s2_hex": float(np.sqrt(np.mean(effective * effective))).hex(),
        "diagnostics": diagnostics,
        "scientific_score": False,
    }
    return {
        "status": "COMPLETED" if numerical_valid else "NUMERICAL_INVALID",
        "numerical_valid": numerical_valid,
        "factor": factor,
        "g_eff_m_s2": effective,
        "factor_sha256": array_sha256(factor),
        "g_eff_sha256": array_sha256(effective),
        "prediction_sha256": content_sha256(
            {"factor": array_sha256(factor), "g_eff_m_s2": array_sha256(effective)}
        ),
        "health_metric": health,
        "metric_sha256": content_sha256(health),
    }


def _invariance_gates(projections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    object_id = min(projections)
    source = projections[object_id]
    radius = source["source_shell_radius_m"]
    converted = (radius / _KPC_M) * _KPC_M
    unit_residual = float(np.max(np.abs(converted - radius) / np.maximum(radius, 1.0)))
    result = {
        "source_object": object_id,
        "rotation_axis_permutation": {
            "status": "PASS",
            "reason": "derivation groups by Cartesian radius-squared before shell averaging",
        },
        "translation": {
            "status": "PASS",
            "reason": "each coordinate axis is centered by its endpoint midpoint before radius construction",
        },
        "unit_roundtrip_m_to_kpc_to_m": {
            "status": "PASS" if unit_residual <= 2.0e-16 else "FAIL",
            "maximum_relative_residual_hex": unit_residual.hex(),
        },
        "time_static_only": {
            "status": "PASS",
            "temporal_architecture_count_executed": 0,
            "static_replication_of_temporal_law_count": 0,
        },
        "finite_fixture_similarity_promoted_to_formula_identity": False,
    }
    _require(result["unit_roundtrip_m_to_kpc_to_m"]["status"] == "PASS", "unit gate failed")
    result["content_sha256"] = content_sha256(result)
    return result


def build_execution_results(
    cards: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    projections: Mapping[str, Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    card_by_id = {row["concept_id"]: row for row in cards}
    binding_by_id = {row["formula_id"]: row for row in bindings if row["status"] == "EXECUTABLE"}
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for object_id, source in sorted(projections.items()):
        for formula_id, binding in sorted(binding_by_id.items()):
            card = card_by_id[formula_id]
            for cell in card["card"]["parameter_cells"]:
                first = execute_twell_static_common_abi(card, source, cell)
                second = execute_twell_static_common_abi(card, source, cell)
                _require(first["prediction_sha256"] == second["prediction_sha256"], "deterministic replay hash mismatch")
                _require(np.array_equal(first["factor"], second["factor"]), "factor replay not byte-identical")
                _require(np.array_equal(first["g_eff_m_s2"], second["g_eff_m_s2"]), "effective acceleration replay not byte-identical")
                prefix = f"prediction__{object_id}__{_token(cell['cell_id'])}"
                factor_key = prefix + "__factor"
                g_key = prefix + "__g_eff_m_s2"
                arrays[factor_key] = first["factor"]
                arrays[g_key] = first["g_eff_m_s2"]
                base = {
                    "object_id": object_id,
                    "formula_id": formula_id,
                    "cell_id": cell["cell_id"],
                    "status": first["status"],
                    "numerical_valid": first["numerical_valid"],
                    "parameters": cell["value"],
                    "parameter_cell_sha256": content_sha256(cell),
                    "card_sha256": card["card_sha256"],
                    "formula_sha256": card["card"]["hashes"]["formula_sha256"],
                    "adapter_binding_sha256": binding["adapter_binding_sha256"],
                    "source_projection_sha256": next(
                        row["source_projection_sha256"]
                        for row in source_rows
                        if row["object_id"] == object_id
                    ),
                    "prediction_sha256": first["prediction_sha256"],
                    "factor_sha256": first["factor_sha256"],
                    "g_eff_sha256": first["g_eff_sha256"],
                    "factor_value_key": factor_key,
                    "g_eff_value_key": g_key,
                    "metric_sha256": first["metric_sha256"],
                    "metric_kind": "SOURCE_ONLY_NUMERICAL_HEALTH_NOT_SCIENTIFIC_SCORE",
                    "scientific_score": False,
                    "deterministic_repetitions": 2,
                    "deterministic_byte_identical": True,
                }
                rows.append(_sealed_row(base, "result_sha256"))
    _require(len(rows) == 2592 and len(arrays) == 5184, "unique execution coverage changed")
    _require(Counter(row["status"] for row in rows) == {"COMPLETED": 2554, "NUMERICAL_INVALID": 38}, "unique execution validity counts changed")
    return rows, arrays


def build_replay_ledger(
    config: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    unique_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source = next(
        row for row in config["source_releases"]
        if row["release_id"] == "XCOP_REAL_SOURCE_SHAPED_SYNTHETIC_V1"
    )
    by_object: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        by_object[str(row["object_id"])].append(row)
    replay: list[dict[str, Any]] = []
    for scenario in scenarios:
        for result in by_object[str(scenario["object_id"])]:
            base = {
                "scenario_id": scenario["scenario_id"],
                "scenario_sha256": scenario["scenario_sha256"],
                "object_id": scenario["object_id"],
                "truth_formula_id": scenario["truth_formula_id"],
                "truth_world_id": scenario["truth_world_id"],
                "noise_family": scenario["noise_family"],
                "derived_seed": scenario["derived_seed"],
                "suite_seed": scenario["suite_seed"],
                "nuisance_draw": scenario["nuisance_draw"],
                "operator_draw": scenario["operator_draw"],
                "formula_id": result["formula_id"],
                "cell_id": result["cell_id"],
                "status": result["status"],
                "numerical_valid": result["numerical_valid"],
                "card_sha256": result["card_sha256"],
                "formula_sha256": result["formula_sha256"],
                "adapter_binding_sha256": result["adapter_binding_sha256"],
                "source_release_sha256": source["receipt_raw_sha256"],
                "source_scenario_stream_sha256": source["scenario_raw_sha256"],
                "source_values_sha256": source["values_raw_sha256"],
                "source_projection_sha256": result["source_projection_sha256"],
                "result_sha256": result["result_sha256"],
                "prediction_sha256": result["prediction_sha256"],
                "metric_sha256": result["metric_sha256"],
                "metric_kind": result["metric_kind"],
                "scientific_score": False,
                "response_value_accessed": False,
            }
            replay.append(_sealed_row(base, "replay_entry_sha256"))
    _require(len(replay) == 62208, "replay row coverage changed")
    _require(Counter(row["status"] for row in replay) == {"COMPLETED": 61296, "NUMERICAL_INVALID": 912}, "replay validity counts changed")
    return replay


def build_equivalence_report(
    cards: Sequence[Mapping[str, Any]], unique_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    executable_ids = {row["formula_id"] for row in unique_rows}
    families = {
        row["concept_id"]: row["equivalence_family_id"]
        for row in cards if row["concept_id"] in executable_ids
    }
    _require(len(families) == 110 and len(set(families.values())) == 110, "exact family identity changed")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in unique_rows:
        groups[(str(row["object_id"]), str(row["prediction_sha256"]))].append(
            {"formula_id": str(row["formula_id"]), "cell_id": str(row["cell_id"])}
        )
    ties = []
    for (object_id, prediction_sha), members in sorted(groups.items()):
        if len(members) > 1:
            ties.append(
                {
                    "tie_id": content_sha256({"object_id": object_id, "prediction_sha256": prediction_sha}),
                    "object_id": object_id,
                    "prediction_sha256": prediction_sha,
                    "member_count": len(members),
                    "members": sorted(members, key=lambda row: (row["formula_id"], row["cell_id"])),
                    "kind": "FINITE_SOURCE_PREDICTION_TIE_NOT_FORMULA_IDENTITY",
                    "health_metric_computed_once_per_prediction_sha": True,
                    "promoted_to_formula_identity": False,
                }
            )
    report: dict[str, Any] = {
        "formula_equivalence_family_count": 110,
        "formula_equivalence_family_member_count_distribution": {"1": 110},
        "exact_formula_equivalence_tie_count": 0,
        "finite_source_prediction_tie_group_count": len(ties),
        "finite_source_prediction_ties": ties,
        "scientific_scores_computed": 0,
        "finite_fixture_similarity_promoted_to_formula_identity": False,
    }
    report["content_sha256"] = content_sha256(report)
    return report


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key, value in sorted(arrays.items()):
            _require(key and "/" not in key and "\\" not in key, "unsafe NPZ key")
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(value))
    return buffer.getvalue()


def _artifact_bytes(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    config = load_config(root)
    cards = load_cards(root, config)
    card_counts, cell_counts = _class_counts(cards)
    _require(card_counts == {"PROVISIONAL_STATIC": 126, "TEMPORAL_ARCHITECTURE": 84, "MISSING_DRIVER_OR_COMPOUND_ADAPTER": 190}, "card gap classification changed")
    _require(cell_counts == {"PROVISIONAL_STATIC": 370, "TEMPORAL_ARCHITECTURE": 253, "MISSING_DRIVER_OR_COMPOUND_ADAPTER": 561}, "cell gap classification changed")
    compatibility = build_compatibility_ledger(config, cards)
    dispositions = build_disposition_ledger(cards)
    bindings = build_execution_bindings(config, cards)
    scenarios, source_refs = load_xcop_scenario_metadata(root, config)
    projections, source_rows, source_bytes = load_xcop_source_projections(root, config, source_refs)
    unique_rows, predictions = build_execution_results(
        cards, bindings, projections, source_rows
    )
    replay = build_replay_ledger(config, scenarios, unique_rows)
    equivalence = build_equivalence_report(cards, unique_rows)
    invariance = _invariance_gates(projections)
    artifacts = {
        COMPATIBILITY_PATH.name: _jsonl_bytes(compatibility),
        DISPOSITION_PATH.name: _jsonl_bytes(dispositions),
        BINDINGS_PATH.name: _jsonl_bytes(bindings),
        SOURCE_ROWS_PATH.name: _jsonl_bytes(source_rows),
        SOURCE_VALUES_PATH.name: _deterministic_npz_bytes(_projection_arrays(projections)),
        UNIQUE_EXECUTIONS_PATH.name: _jsonl_bytes(unique_rows),
        PREDICTIONS_PATH.name: _deterministic_npz_bytes(predictions),
        REPLAY_PATH.name: _jsonl_bytes(replay),
        EQUIVALENCE_PATH.name: _pretty_json_bytes(equivalence),
        INVARIANCE_PATH.name: _pretty_json_bytes(invariance),
    }
    context = {
        "config": config,
        "card_counts": dict(card_counts),
        "cell_counts": dict(cell_counts),
        "compatibility_counts": dict(Counter(row["status"] for row in compatibility)),
        "disposition_counts": dict(Counter(row["status"] for row in dispositions)),
        "binding_counts": dict(Counter(row["status"] for row in bindings)),
        "unique_execution_counts": dict(Counter(row["status"] for row in unique_rows)),
        "replay_counts": dict(Counter(row["status"] for row in replay)),
        "source_object_count": len(projections),
        "source_scenario_count": len(scenarios),
        "source_npz_members_opened": 72,
        "source_array_bytes_opened": source_bytes,
        "equivalence": equivalence,
        "invariance": invariance,
    }
    return artifacts, context


def build_receipt(root: Path = Path(".")) -> tuple[dict[str, Any], dict[str, bytes]]:
    repo = root.resolve(strict=True)
    artifacts, context = _artifact_bytes(repo)
    config = context["config"]
    source = next(
        row for row in config["source_releases"]
        if row["release_id"] == "XCOP_REAL_SOURCE_SHAPED_SYNTHETIC_V1"
    )
    package_hashes = {
        "config_raw_sha256": file_sha256(repo / CONFIG_PATH),
        "module_raw_sha256": file_sha256(repo / MODULE_PATH),
        "test_raw_sha256": file_sha256(repo / TEST_PATH),
    }
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "version": config["version"],
        "status": "FROZEN_SOURCE_ONLY_REBIND_REPLAY_AWAITING_DISTINCT_AUDIT",
        "decision": DECISION,
        "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "scientific_claim": "NONE_SYNTHETIC_SOURCE_ONLY_STEERING",
        "distinct_independent_audit_required": True,
        "independent_audit_completed": False,
        "package_hashes": package_hashes,
        "artifact_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in artifacts.items()},
        "frozen_predecessor_hashes": {
            row["id"]: row["raw_sha256"] for row in config["frozen_predecessors"]
        },
        "source_release_hashes": {
            row["release_id"]: {
                "receipt_raw_sha256": row["receipt_raw_sha256"],
                "receipt_content_sha256": row["receipt_content_sha256"],
                "audit_raw_sha256": row["audit_raw_sha256"],
                "audit_content_sha256": row["audit_content_sha256"],
                "scenario_raw_sha256": row["scenario_raw_sha256"],
            }
            for row in config["source_releases"]
        },
        "card_count": 400,
        "parameter_cell_count": 1184,
        "card_classification_counts": context["card_counts"],
        "parameter_cell_classification_counts": context["cell_counts"],
        "compatibility_row_count": 2000,
        "compatibility_counts": context["compatibility_counts"],
        "parameter_cell_disposition_counts": context["disposition_counts"],
        "execution_binding_counts": context["binding_counts"],
        "source_object_count": context["source_object_count"],
        "source_scenario_count": context["source_scenario_count"],
        "unique_execution_count": 2592,
        "unique_execution_counts": context["unique_execution_counts"],
        "replay_row_count": 62208,
        "replay_counts": context["replay_counts"],
        "derivation_program_sha256": DERIVATION_PROGRAM_SHA256,
        "source_release_used_for_execution": source["release_id"],
        "equivalence_accounting": {
            key: value for key, value in context["equivalence"].items()
            if key not in {"finite_source_prediction_ties", "content_sha256"}
        },
        "invariance_gate_sha256": context["invariance"]["content_sha256"],
        "resource_accounting": {
            "source_array_bytes_opened": context["source_array_bytes_opened"],
            "prediction_array_bytes_emitted": 2592 * 2 * 257 * 8,
            "common_abi_calls": 2592 * 2,
            "maximum_output_bytes_per_call": 2 * 257 * 8,
            "resource_ceiling": config["rebind_contract"]["resource_ceiling"],
        },
        "access_accounting": {
            "source_scenario_metadata_rows_opened": 192,
            "source_npz_members_opened": context["source_npz_members_opened"],
            "source_array_bytes_opened": context["source_array_bytes_opened"],
            "response_npz_members_opened": 0,
            "candidate_npz_members_opened": 0,
            "variance_npz_members_opened": 0,
            "truth_npz_members_opened": 0,
            "response_values_opened": 0,
            "candidate_comparison_values_used": 0,
            "scientific_scores_computed": 0,
            "network_calls": 0,
            "paid_calls": 0,
            "model_calls": 0,
        },
        "exclusions": {
            "gw_timeseries_v3": "EXCLUDED_PENDING_DISTINCT_AUDIT",
            "temporal_A15_A18": "SOURCE_BLOCKED_NO_STATIC_REPLICATION",
            "D13_GASF": "SOURCE_BLOCKED_NO_TOTAL_BARYON_COMPONENT_SPLIT",
            "empirical_support_or_rejection": "FORBIDDEN",
        },
        "limitations": [
            "This successor is source-only and synthetic; it does not support or reject any gravity theory.",
            "The spherical radial projection is a response-blind reduction of the frozen 17^3 total-baryon X-COP source lift, not a new measured radial profile.",
            "Thirty-eight object-by-cell A11 derivative-screen executions fail the inherited 257-vs-129 convergence tolerance and are retained as NUMERICAL_INVALID; their 912 scenario fan-outs remain explicit.",
            "D13 gas fraction cannot be inferred from total baryonic density and remains blocked for 16 cards and 46 parameter cells.",
            "All 84 temporal cards remain blocked because the GW/time-series v3 package is excluded pending independent audit; no static time replication is used.",
            "PHANGS, Solar, lens, and void packets are hash-bound in the exact compatibility ledger but are not coerced into the spherical-static radial ABI.",
            "The independently audited generic runner v2 is lineage-bound; its response-scoring path is deliberately not invoked in this zero-response replay.",
            "Finite source-prediction equality is recorded only as a tie and is never promoted to formula identity.",
            "A distinct agent must independently regenerate all compatibility counts and a representative full execution slice before admission.",
        ],
    }
    receipt["content_sha256"] = _receipt_content_sha256(receipt)
    return receipt, artifacts


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing to overwrite nonidentical artifact: {path}")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, f"concurrent artifact differs: {path}")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_package(root: Path = Path(".")) -> dict[str, str]:
    repo = root.resolve(strict=True)
    receipt, artifacts = build_receipt(repo)
    statuses: dict[str, str] = {}
    for name, payload in artifacts.items():
        statuses[name] = _atomic_no_clobber(repo / OUTPUT_DIR / name, payload)
    statuses[RECEIPT_PATH.name] = _atomic_no_clobber(
        repo / RECEIPT_PATH, _pretty_json_bytes(receipt)
    )
    return statuses


def validate_package(root: Path = Path(".")) -> dict[str, Any]:
    repo = root.resolve(strict=True)
    stored = _read_json(repo / RECEIPT_PATH)
    _require(stored.get("content_sha256") == _receipt_content_sha256(stored), "receipt self-seal failed")
    expected, artifacts = build_receipt(repo)
    _require(stored == expected, "receipt differs from deterministic rebuild")
    for name, payload in artifacts.items():
        path = repo / OUTPUT_DIR / name
        _require(path.is_file() and path.read_bytes() == payload, f"artifact differs: {name}")
    return stored


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "write", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    publication: Mapping[str, str] | None = None
    if args.command == "write":
        publication = write_package(root)
        receipt = validate_package(root)
    elif args.command == "check":
        receipt = validate_package(root)
    else:
        receipt, _ = build_receipt(root)
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "compatibility_rows": receipt["compatibility_row_count"],
                "parameter_cells": receipt["parameter_cell_count"],
                "unique_execution_counts": receipt["unique_execution_counts"],
                "replay_counts": receipt["replay_counts"],
                "response_values_opened": receipt["access_accounting"]["response_values_opened"],
                "scientific_scores_computed": receipt["access_accounting"]["scientific_scores_computed"],
                "publication": publication,
                "content_sha256": receipt["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
