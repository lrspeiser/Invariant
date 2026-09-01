"""Response-blind CF4/VAST source-shaped void/cosmology synthetic matrix."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler.open_gravity_data_element_ontology_v1 import (
    Availability,
    DataElement,
    DataRole,
    ExperimentRole,
    UncertaintyKind,
    catalogue_from_elements,
)
from sigma_theory_compiler.open_gravity_formula_adapter_registry_v1 import (
    AdapterRegistration,
    validate_adapter_registry,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    FormulaExecutionBinding,
    ResourceBounds,
    validate_binding_catalogue,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v1 import (
    ObservableComparison,
    ParameterCell,
    ScenarioRuntimeValues,
)
from sigma_theory_compiler.open_gravity_synthetic_discovery_runner_v2 import (
    run_discovery_matrix_v2,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    SyntheticSuiteRelease,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    AnchorBinding,
    AxisSpec,
    EmittedPredictionSpec,
    FeatureValueRef,
    ScenarioDescriptor,
    UncertaintyRef,
    array_sha256,
)
from sigma_theory_compiler.open_gravity_void_geometry_source_completion_v2 import (
    mask_contains,
    mask_index,
)
from sigma_theory_compiler.open_gravity_void_geometry_source_completion_v3 import (
    luminosity_to_comoving_hinv,
    radec_to_xyz,
    validate_cf4_distance,
)
from sigma_theory_compiler.open_gravity_void_gravitational_load_v4 import (
    ray_sphere_intervals,
    union_intervals,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path(
    "configs/open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v1.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v1.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v1"
)
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
DIAGNOSTICS_PATH = OUTPUT_DIR / "geometry-and-identifiability.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]

_FEATURES = tuple(
    sorted(
        (
            "source.scalar.delta-h-km-s-mpc",
            "source.scalar.distance-modulus-mag",
            "source.scalar.distance-modulus-uncertainty-mag",
            "source.scalar.distance-mpc",
            "source.scalar.h-m-km-s-mpc",
            "source.scalar.mask-neighborhood-fraction",
            "source.scalar.maximum-chord-mpc",
            "source.scalar.null-void-length-mpc",
            "source.scalar.observer-endpoint-chord-mpc",
            "source.scalar.target-endpoint-chord-mpc",
            "source.scalar.void-fraction",
            "source.scalar.void-length-mpc",
            "source.vector.direction-cartesian",
            "source.vector.flow-shear-design",
        )
    )
)
_OUTPUT = "prediction.scalar.log-redshift"
_RESPONSE = "response.synthetic-log-redshift"
_UNCERTAINTY = "uncertainty.response.synthetic-log-redshift"
_TRUTH = "truth.scalar.injection-id"
_EXECUTABLE = {
    "C01_OBSERVER_ENDPOINT_LOCAL_VOID": "observer_endpoint_adapter",
    "C02_TARGET_ENDPOINT_LOCAL_VOID": "target_endpoint_adapter",
    "C03_SINGLE_DOMINANT_VOID": "maximum_chord_adapter",
    "C04_BOUNDED_FRACTION_NULL": "bounded_fraction_null_adapter",
    "VQ00_STANDARD_FLRW_FLOW_CONTROL": "standard_flrw_adapter",
    "VQ08_TWO_PHASE_VOID_FRACTION": "two_phase_void_adapter",
}
_BLOCKED = {
    "VQ01_DIRECT_TIME_EXPOSURE",
    "VQ02_SLOWED_RAY_TIME_EXPOSURE",
    "VQ03_LOCAL_SOURCE_SINK",
    "VQ04_3D_HELMHOLTZ_BARYON_FEED",
    "VQ05_3D_DIFFUSIVE_RESERVOIR",
    "VQ06_3D_COLUMN_ATTENUATED_FEED",
    "VQ07_INVERSE_DENSITY_HUBBLE_MIMIC",
    "VQ09_PHOTON_CARRIED_MEMORY",
    "VQ10_TIED_RESERVOIR_DELAY_REDSHIFT",
}
_C_KM_S = 299792.458


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaViolation(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _repo_path(value: str | Path) -> Path:
    parsed = PurePosixPath(str(value).replace("\\", "/"))
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise SchemaViolation("void synthetic path escaped repository")
    result = (_ROOT / parsed.as_posix()).resolve()
    _require(result.is_relative_to(_ROOT), "void synthetic path escaped repository")
    return result


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    expected = {
        "schema",
        "package_id",
        "version",
        "status",
        "claim_class",
        "experiment_id",
        "suite_seed",
        "object_count",
        "geometry_variants",
        "noise_families",
        "truth_formula_ids",
        "law_constants",
        "noise",
        "scoring",
        "geometry_mode",
        "time_mode",
        "coordinate_frame",
        "parameter_schema_path",
        "output_directory",
        "source_anchors",
        "contract_bindings",
        "adapter_blocks",
        "access_contract",
    }
    _require(set(config) == expected, "void synthetic config keys changed")
    _require(
        config["schema"]
        == "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-1.0",
        "void synthetic schema changed",
    )
    _require(
        config["package_id"]
        == "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v1"
        and config["version"] == "v1.0.0",
        "void synthetic identity changed",
    )
    _require(config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE", "status changed")
    _require(config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL", "claim changed")
    _require(
        config["object_count"] == 8
        and config["geometry_variants"]
        == [
            "bounded-fraction-null",
            "planck2018-edge0-primary",
            "wmap5-edge0-control",
        ]
        and len(config["noise_families"]) == 5,
        "matrix axes changed",
    )
    _require(config["truth_formula_ids"] == sorted(_EXECUTABLE), "truth formulas changed")
    block_ids = [row["formula_id"] for row in config["adapter_blocks"]]
    _require(block_ids == sorted(_BLOCKED), "blocked VQ catalogue changed")
    _require(
        float(config["law_constants"]["c_km_s"]) == _C_KM_S
        and float(config["law_constants"]["h_m_km_s_mpc"]) == 67.4
        and float(config["law_constants"]["delta_h_km_s_mpc"]) == 6.74,
        "law constants changed",
    )
    _require(
        _repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(),
        "output path changed",
    )
    access = config["access_contract"]
    zero_keys = (
        "cf4_measured_velocity_fields_decoded",
        "cf4_published_peculiar_velocity_fields_decoded",
        "confirmation_source_fields_decoded",
        "model_calls",
        "network_calls",
        "paid_calls",
        "pantheon_files_opened",
        "real_response_values_decoded",
        "theory_or_nuisance_response_tuning_events",
        "validation_source_fields_decoded",
    )
    _require(all(access[key] == 0 for key in zero_keys), "response access boundary changed")
    _require(
        access["cf4_allowed_source_fields"] == ["1PGC", "DEdeg", "DMzp", "Dist", "RAdeg", "e_DMzp"],
        "CF4 source field allowlist changed",
    )
    for group in (config["source_anchors"], config["contract_bindings"]):
        ids = [row["id"] for row in group]
        _require(ids == sorted(set(ids)), "source/contract IDs must be sorted unique")
        for row in group:
            path = _repo_path(row["path"])
            if verify_hashes:
                _require(path.is_file(), f"missing binding: {row['id']}")
                _require(_file_sha256(path) == row["sha256"], f"binding drift: {row['id']}")
                if "bytes" in row:
                    _require(
                        path.stat().st_size == row["bytes"], f"binding size drift: {row['id']}"
                    )
    if verify_hashes:
        schema_path = _repo_path(config["parameter_schema_path"])
        _require(schema_path.is_file(), "parameter schema missing")


def _split_role(identifier: int) -> tuple[int, str]:
    _require(type(identifier) is int and identifier > 0, "invalid CF4 identifier")
    bucket = (
        int.from_bytes(hashlib.sha256(str(identifier).encode("ascii")).digest()[:8], "big") % 10
    )
    role = "development" if bucket <= 5 else "validation" if bucket <= 7 else "confirmation"
    return bucket, role


def _load_identifier_ledger(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    row = next(
        item for item in config["source_anchors"] if item["id"] == "CF4_IDENTIFIER_ROLE_LEDGER"
    )
    entries = [
        json.loads(line)
        for line in _repo_path(row["path"]).read_text(encoding="utf-8").splitlines()
    ]
    _require(len(entries) == 38053, "identifier ledger row count changed")
    _require(
        [int(item["source_index"]) for item in entries] == list(range(len(entries))),
        "identifier ledger order changed",
    )
    return entries


def _parse_float_slice(payload: bytes, start: int, stop: int, label: str) -> float:
    try:
        value = float(payload[start:stop].decode("ascii").strip())
    except (UnicodeDecodeError, ValueError) as error:
        raise SchemaViolation(f"invalid permitted CF4 {label}") from error
    _require(math.isfinite(value), f"nonfinite permitted CF4 {label}")
    return value


def _parse_permitted_cf4_source(payload: bytes, identifier: int) -> dict[str, float | int]:
    """Decode only source fields; bytes 39:44 (V3k) are never sliced or parsed."""

    dmzp = _parse_float_slice(payload, 8, 14, "DMzp")
    e_dmzp = _parse_float_slice(payload, 15, 20, "e_DMzp")
    distance = _parse_float_slice(payload, 21, 26, "Dist")
    ra = _parse_float_slice(payload, 83, 91, "RAdeg")
    dec = _parse_float_slice(payload, 92, 100, "DEdeg")
    _require(e_dmzp > 0.0 and distance > 0.0, "invalid CF4 source uncertainty or distance")
    _require(0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0, "invalid CF4 direction")
    validate_cf4_distance(dmzp, distance)
    return {
        "identifier": identifier,
        "DMzp": dmzp,
        "e_DMzp": e_dmzp,
        "Dist": distance,
        "RAdeg": ra,
        "DEdeg": dec,
    }


def _parse_vast_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    anchors = {row["id"]: row for row in config["source_anchors"]}
    edge_by_key: dict[tuple[str, int], int] = {}
    table1_path = _repo_path(anchors["VAST1_MAXIMAL_SPHERES_AND_EDGE_FLAGS"]["path"])
    table1_count = 0
    for raw in table1_path.read_bytes().splitlines():
        parts = raw.decode("ascii").split()
        _require(len(parts) == 11, "invalid VAST1 source row")
        cosmology = parts[0]
        void_id = int(parts[5])
        edge = int(parts[6])
        _require(cosmology in {"Planck2018", "WMAP5"}, "invalid VAST1 cosmology")
        _require(void_id >= 0 and edge in {0, 1, 2}, "invalid VAST1 identifiers")
        key = (cosmology, void_id)
        _require(key not in edge_by_key, "duplicate VAST1 void key")
        edge_by_key[key] = edge
        table1_count += 1
    _require(table1_count == 2347, "VAST1 row count changed")

    groups: dict[str, list[tuple[tuple[float, float, float], float]]] = {
        "Planck2018": [],
        "WMAP5": [],
    }
    table2_path = _repo_path(anchors["VAST2_ALL_SPHERE_UNION_GEOMETRY"]["path"])
    table2_count = 0
    observed_groups: set[tuple[str, int]] = set()
    semantic_keys: set[tuple[str, int, str, str, str, str]] = set()
    with gzip.open(table2_path, "rb") as handle:
        for raw in handle:
            payload = raw.rstrip(b"\n")
            if payload.endswith(b"\r"):
                payload = payload[:-1]
            _require(len(payload) == 105, "invalid VAST2 framed payload")
            parts = payload.decode("ascii").split()
            _require(len(parts) == 6, "invalid VAST2 source row")
            cosmology = parts[0]
            x, y, z, radius = (float(value) for value in parts[1:5])
            void_id = int(parts[5])
            key = (cosmology, void_id)
            _require(key in edge_by_key and radius > 0.0, "invalid VAST2 group or radius")
            semantic = (cosmology, void_id, *parts[1:5])
            _require(semantic not in semantic_keys, "duplicate VAST2 sphere")
            semantic_keys.add(semantic)
            observed_groups.add(key)
            if edge_by_key[key] == 0:
                h = float(config["law_constants"]["planck_h"])
                groups[cosmology].append(((x / h, y / h, z / h), radius / h))
            table2_count += 1
    _require(table2_count == 80080, "VAST2 row count changed")
    _require(observed_groups == set(edge_by_key), "VAST1/VAST2 group mismatch")
    _require(all(groups[name] for name in groups), "empty retained VAST geometry")
    return {
        "spheres": groups,
        "table1_rows": table1_count,
        "table2_rows": table2_count,
        "retained": {name: len(values) for name, values in groups.items()},
    }


def _interval_summary(
    direction: np.ndarray,
    distance: float,
    spheres: Sequence[tuple[tuple[float, float, float], float]],
) -> dict[str, float | int]:
    intervals = union_intervals(ray_sphere_intervals(direction, distance, spheres))
    length = math.fsum(stop - start for start, stop in intervals)
    maximum = max((stop - start for start, stop in intervals), default=0.0)
    observer = intervals[0][1] if intervals and intervals[0][0] == 0.0 else 0.0
    target = distance - intervals[-1][0] if intervals and intervals[-1][1] == distance else 0.0
    tolerance = 1e-9 * max(1.0, distance)
    _require(-tolerance <= length <= distance + tolerance, "void path exceeds total path")
    length = min(max(length, 0.0), distance)
    for value in (maximum, observer, target):
        _require(0.0 <= value <= length + tolerance, "chord exceeds void union length")
    return {
        "void_length_mpc": length,
        "void_fraction": length / distance,
        "maximum_chord_mpc": maximum,
        "observer_endpoint_chord_mpc": observer,
        "target_endpoint_chord_mpc": target,
        "crossing_count": len(intervals),
    }


def _mask_neighborhood_fraction(mask: bytes, ra: float, dec: float) -> float:
    i, j = mask_index(ra, dec)
    values = [
        mask[((i + di) % 360) * 180 + min(max(j + dj, 0), 179)]
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
    ]
    return math.fsum(values) / 9.0


def _select_source_objects(
    config: Mapping[str, Any], geometry: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    anchors = {row["id"]: row for row in config["source_anchors"]}
    mask = _repo_path(anchors["CANONICAL_VAST_ANGULAR_MASK"]["path"]).read_bytes()
    _require(len(mask) == 64800 and set(mask) <= {0, 1}, "canonical mask invalid")
    ledger = _load_identifier_ledger(config)
    source_path = _repo_path(anchors["CF4_TABLE4_OPAQUE_ROW_CONTAINER"]["path"])
    selected: list[dict[str, Any]] = []
    source_fields_decoded = 0
    raw_rows_read = 0
    stream_offset = 0
    radial_limit = float(config["law_constants"]["radial_mask_limit_h_inverse_mpc"])
    h = float(config["law_constants"]["planck_h"])
    with gzip.open(source_path, "rb") as handle:
        for source_index, raw in enumerate(handle):
            entry = ledger[source_index]
            raw_rows_read += 1
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                payload = payload[:-1]
            _require(len(payload) == 157, "invalid CF4 payload length")
            _require(int(entry["framed_start"]) == stream_offset, "CF4 ledger offset mismatch")
            _require(
                hashlib.sha256(raw).hexdigest() == entry["framed_raw_sha256"]
                and hashlib.sha256(payload).hexdigest() == entry["payload_raw_sha256"],
                "CF4 ledger row hash mismatch",
            )
            stream_offset += len(raw)
            identifier = int(payload[:7].decode("ascii").strip())
            bucket, role = _split_role(identifier)
            _require(
                identifier == int(entry["identifier"])
                and bucket == int(entry["bucket"])
                and role == entry["role"],
                "CF4 identifier partition mismatch",
            )
            if role != "development":
                continue
            source = _parse_permitted_cf4_source(payload, identifier)
            source_fields_decoded += 1
            ra, dec = float(source["RAdeg"]), float(source["DEdeg"])
            if not mask_contains(mask, ra, dec):
                continue
            _, distance_hinv = luminosity_to_comoving_hinv(float(source["Dist"]))
            if not 0.0 < distance_hinv <= radial_limit:
                continue
            distance = distance_hinv / h
            direction = radec_to_xyz(ra, dec, 1.0)
            planck = _interval_summary(direction, distance, geometry["spheres"]["Planck2018"])
            wmap = _interval_summary(direction, distance, geometry["spheres"]["WMAP5"])
            selected.append(
                {
                    **source,
                    "source_index": source_index,
                    "bucket": bucket,
                    "role": role,
                    "distance_path_mpc": distance,
                    "direction": tuple(float(value) for value in direction),
                    "mask_neighborhood_fraction": _mask_neighborhood_fraction(mask, ra, dec),
                    "planck": planck,
                    "wmap": wmap,
                }
            )
            if len(selected) == int(config["object_count"]):
                break
    _require(
        len(selected) == int(config["object_count"]), "insufficient source-eligible CF4 objects"
    )
    _require(len({row["identifier"] for row in selected}) == len(selected), "duplicate CF4 object")
    return selected, {
        "cf4_raw_rows_read": raw_rows_read,
        "cf4_development_source_rows_decoded": source_fields_decoded,
        "cf4_measured_velocity_fields_decoded": 0,
        "cf4_published_peculiar_velocity_fields_decoded": 0,
        "validation_source_fields_decoded": 0,
        "confirmation_source_fields_decoded": 0,
    }


def _fraction_permutation(rows: Sequence[Mapping[str, Any]], key: str) -> dict[int, float]:
    ordered = sorted(
        rows, key=lambda row: (float(row["distance_path_mpc"]), int(row["identifier"]))
    )
    result: dict[int, float] = {}
    for start in range(0, len(ordered), 4):
        stratum = ordered[start : start + 4]
        fractions = [float(row[key]["void_fraction"]) for row in stratum]
        rotated = fractions[1:] + fractions[:1]
        for row, fraction in zip(stratum, rotated, strict=True):
            _require(0.0 <= fraction <= 1.0, "permuted void fraction outside unit interval")
            result[int(row["identifier"])] = fraction
    _require(
        set(result) == {int(row["identifier"]) for row in rows}, "fraction permutation incomplete"
    )
    return result


def _variant_items(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    permutations = {
        "planck": _fraction_permutation(rows, "planck"),
        "wmap": _fraction_permutation(rows, "wmap"),
    }
    items: list[dict[str, Any]] = []
    for row in rows:
        identifier = int(row["identifier"])
        distance = float(row["distance_path_mpc"])
        for variant in config["geometry_variants"]:
            if variant == "planck2018-edge0-primary":
                geometry = dict(row["planck"])
                null_length = permutations["planck"][identifier] * distance
                source_geometry = "planck"
            elif variant == "wmap5-edge0-control":
                geometry = dict(row["wmap"])
                null_length = permutations["wmap"][identifier] * distance
                source_geometry = "wmap"
            else:
                original = dict(row["planck"])
                permuted_length = permutations["planck"][identifier] * distance
                scale = (
                    permuted_length / float(original["void_length_mpc"])
                    if float(original["void_length_mpc"]) > 0.0
                    else 0.0
                )
                geometry = {
                    "void_length_mpc": permuted_length,
                    "void_fraction": permuted_length / distance,
                    "maximum_chord_mpc": min(
                        permuted_length, float(original["maximum_chord_mpc"]) * scale
                    ),
                    "observer_endpoint_chord_mpc": min(
                        permuted_length, float(original["observer_endpoint_chord_mpc"]) * scale
                    ),
                    "target_endpoint_chord_mpc": min(
                        permuted_length, float(original["target_endpoint_chord_mpc"]) * scale
                    ),
                    "crossing_count": int(original["crossing_count"]),
                }
                null_length = float(original["void_length_mpc"])
                source_geometry = "planck-fraction-permuted-within-distance-stratum"
            length = float(geometry["void_length_mpc"])
            _require(0.0 <= length <= distance, "variant void length outside total distance")
            _require(0.0 <= null_length <= distance, "null void length outside total distance")
            direction = np.asarray(row["direction"], dtype=np.float64)
            dx, dy, dz = (float(value) for value in direction)
            design = np.asarray(
                [dx, dy, dz, dx * dx, dy * dy, dz * dz, dx * dy, dx * dz, dy * dz],
                dtype=np.float64,
            )
            values = {
                "source.scalar.delta-h-km-s-mpc": np.asarray(
                    [config["law_constants"]["delta_h_km_s_mpc"]], dtype=np.float64
                ),
                "source.scalar.distance-modulus-mag": np.asarray([row["DMzp"]], dtype=np.float64),
                "source.scalar.distance-modulus-uncertainty-mag": np.asarray(
                    [row["e_DMzp"]], dtype=np.float64
                ),
                "source.scalar.distance-mpc": np.asarray([distance], dtype=np.float64),
                "source.scalar.h-m-km-s-mpc": np.asarray(
                    [config["law_constants"]["h_m_km_s_mpc"]], dtype=np.float64
                ),
                "source.scalar.mask-neighborhood-fraction": np.asarray(
                    [row["mask_neighborhood_fraction"]], dtype=np.float64
                ),
                "source.scalar.maximum-chord-mpc": np.asarray(
                    [geometry["maximum_chord_mpc"]], dtype=np.float64
                ),
                "source.scalar.null-void-length-mpc": np.asarray([null_length], dtype=np.float64),
                "source.scalar.observer-endpoint-chord-mpc": np.asarray(
                    [geometry["observer_endpoint_chord_mpc"]], dtype=np.float64
                ),
                "source.scalar.target-endpoint-chord-mpc": np.asarray(
                    [geometry["target_endpoint_chord_mpc"]], dtype=np.float64
                ),
                "source.scalar.void-fraction": np.asarray([length / distance], dtype=np.float64),
                "source.scalar.void-length-mpc": np.asarray([length], dtype=np.float64),
                "source.vector.direction-cartesian": direction,
                "source.vector.flow-shear-design": design,
            }
            _require(set(values) == set(_FEATURES), "variant feature set changed")
            items.append(
                {
                    "identifier": identifier,
                    "source_index": int(row["source_index"]),
                    "variant": variant,
                    "source_geometry": source_geometry,
                    "values": values,
                    "distance_mpc_hex": distance.hex(),
                    "void_length_mpc_hex": length.hex(),
                    "null_void_length_mpc_hex": null_length.hex(),
                    "void_fraction_hex": (length / distance).hex(),
                }
            )
    return sorted(items, key=lambda row: (int(row["identifier"]), str(row["variant"])))


def _catalogue(config: Mapping[str, Any]):
    provenance = canonical_sha256(
        {"source": config["source_anchors"], "contracts": config["contract_bindings"]}
    )
    specs = [
        (
            feature,
            "Mpc"
            if feature.endswith("-mpc")
            else "km s^-1 Mpc^-1"
            if "km-s-mpc" in feature
            else "mag"
            if "modulus" in feature
            else "1",
            ("cartesian",)
            if "direction-cartesian" in feature
            else ("nuisance",)
            if "flow-shear-design" in feature
            else ("object",),
            1 if feature.startswith("source.vector") else 0,
        )
        for feature in _FEATURES
    ]
    specs.extend(
        [
            (_OUTPUT, "1", ("object",), 0),
            (_RESPONSE, "1", ("object",), 0),
            (_TRUTH, "integer code", ("object",), 0),
        ]
    )
    dimensions = {
        "1": (0, 0, 0, 0, 0, 0, 0),
        "integer code": (0, 0, 0, 0, 0, 0, 0),
        "mag": (0, 0, 0, 0, 0, 0, 0),
        "Mpc": (0, 1, 0, 0, 0, 0, 0),
        "km s^-1 Mpc^-1": (0, 0, -1, 0, 0, 0, 0),
    }
    elements = []
    for element_id, unit, axes, rank in specs:
        if element_id == _RESPONSE:
            role, availability = DataRole.SCORING_ONLY_RESPONSE, Availability.SYNTHETIC_ONLY
        elif element_id == _OUTPUT:
            role, availability = DataRole.DERIVED, Availability.SYNTHETIC_ONLY
        elif element_id == _TRUTH:
            role, availability = DataRole.LATENT_SYNTHETIC_TRUTH, Availability.SYNTHETIC_ONLY
        else:
            role, availability = DataRole.FORMULA_INPUT, Availability.PUBLIC_SOURCE
        elements.append(
            DataElement(
                element_id=element_id,
                namespace=element_id.rsplit(".", 1)[0],
                physical_quantity=element_id,
                tensor_rank=rank,
                si_dimension=dimensions[unit],
                canonical_unit=unit,
                frame="latent" if element_id == _TRUTH else config["coordinate_frame"],
                support="response-blind CF4/VAST source-shaped synthetic path geometry",
                axes=axes,
                component="total",
                derivation_parents=(),
                uncertainty=UncertaintyKind.COVARIANCE
                if element_id in {_OUTPUT, _RESPONSE}
                else UncertaintyKind.NONE,
                availability=availability,
                experiment_roles=(ExperimentRole(config["experiment_id"], role),),
                provenance_sha256=provenance,
            )
        )
    return catalogue_from_elements(
        "open-gravity-void-cosmology-source-shaped-synthetic", "v1.0.0", elements
    )


def _bindings(config: Mapping[str, Any]) -> tuple[FormulaExecutionBinding, ...]:
    schema_path = _repo_path(config["parameter_schema_path"])
    schema_hash = _file_sha256(schema_path)
    reasons = {row["formula_id"]: row["reason"] for row in config["adapter_blocks"]}
    bindings = []
    for formula_id in sorted((*_EXECUTABLE, *_BLOCKED)):
        executable = formula_id in _EXECUTABLE
        bindings.append(
            FormulaExecutionBinding(
                binding_id=f"binding.void-cosmology.{formula_id.lower()}.v1",
                formula_id=formula_id,
                formula_version="v1.0.0-source-shaped-synthetic",
                formula_sha256=canonical_sha256(
                    {
                        "formula_id": formula_id,
                        "law_contract": "H_m*D+delta_H*registered_exposure over c",
                        "source_block": reasons.get(formula_id),
                    }
                ),
                status=BindingStatus.EXECUTABLE if executable else BindingStatus.SOURCE_BLOCKED,
                entrypoint=(
                    f"sigma_theory_compiler.open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v1:{_EXECUTABLE[formula_id]}"
                    if executable
                    else None
                ),
                required_features=_FEATURES
                if executable
                else ("source.field.unavailable-load-state",),
                optional_features=(),
                emitted_features=(_OUTPUT,),
                domains=("void-cosmology",),
                geometry_support=(config["geometry_mode"],),
                time_support=(config["time_mode"],),
                parameter_schema_path=config["parameter_schema_path"],
                parameter_schema_sha256=schema_hash,
                approximation_ceiling=(
                    "low-redshift source-shaped synthetic log-redshift; no measured velocity/redshift residual"
                    if executable
                    else reasons[formula_id]
                ),
                health_gates=(
                    "bounded-path",
                    "determinism",
                    "finite-output",
                    "source-isolation",
                    "typed-output",
                    "unit",
                ),
                resource_bounds=ResourceBounds(30, 500_000_000, 2_000_000),
            )
        )
    return tuple(bindings)


def _prediction(features: Mapping[str, Any], exposure_key: str | None) -> dict[str, np.ndarray]:
    _require(set(features) == set(_FEATURES), "void synthetic feature projection changed")
    _require(
        not features.get("response") and not features.get("V3k"), "response leaked into adapter"
    )
    distance = float(np.asarray(features["source.scalar.distance-mpc"])[0])
    h_m = float(np.asarray(features["source.scalar.h-m-km-s-mpc"])[0])
    delta_h = float(np.asarray(features["source.scalar.delta-h-km-s-mpc"])[0])
    exposure = 0.0 if exposure_key is None else float(np.asarray(features[exposure_key])[0])
    _require(0.0 <= exposure <= distance and distance > 0.0, "adapter exposure outside path")
    value = (h_m * distance + delta_h * exposure) / _C_KM_S
    _require(math.isfinite(value) and value >= 0.0, "nonfinite void log-redshift")
    return {_OUTPUT: np.asarray([value], dtype=np.float64)}


def standard_flrw_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "VQ00 parameters must be empty")
    return _prediction(features, None)


def two_phase_void_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "VQ08 parameters must be empty")
    return _prediction(features, "source.scalar.void-length-mpc")


def observer_endpoint_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "C01 parameters must be empty")
    return _prediction(features, "source.scalar.observer-endpoint-chord-mpc")


def target_endpoint_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "C02 parameters must be empty")
    return _prediction(features, "source.scalar.target-endpoint-chord-mpc")


def maximum_chord_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "C03 parameters must be empty")
    return _prediction(features, "source.scalar.maximum-chord-mpc")


def bounded_fraction_null_adapter(
    features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    _require(not parameters, "C04 parameters must be empty")
    return _prediction(features, "source.scalar.null-void-length-mpc")


def _noise_response(
    prediction: np.ndarray,
    values: Mapping[str, np.ndarray],
    family: str,
    lineage: SeedLineage,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    rng = np.random.default_rng(lineage.derived_seed)
    direction = np.asarray(values["source.vector.direction-cartesian"], dtype=np.float64)
    design = np.asarray(values["source.vector.flow-shear-design"], dtype=np.float64)
    distance = float(values["source.scalar.distance-mpc"][0])
    e_dm = float(values["source.scalar.distance-modulus-uncertainty-mag"][0])
    h_m = float(values["source.scalar.h-m-km-s-mpc"][0])
    delta_h = float(values["source.scalar.delta-h-km-s-mpc"][0])
    mask_fraction = float(values["source.scalar.mask-neighborhood-fraction"][0])
    sigma_distance = (math.log(10.0) / 5.0) * e_dm * h_m * distance / _C_KM_S
    minimum = float(config["noise"]["minimum_log_redshift_sigma"])
    bulk = np.asarray(config["noise"]["bulk_velocity_coefficients_km_s"], dtype=np.float64)
    shear = np.asarray(config["noise"]["shear_velocity_coefficients_km_s"], dtype=np.float64)
    flow = (float(np.dot(direction, bulk)) + float(np.dot(design[3:], shear))) / _C_KM_S
    distance_draw = float(rng.normal()) * sigma_distance
    boundary = (
        (2.0 * mask_fraction - 1.0)
        * float(config["noise"]["mask_boundary_delta_h_fraction"])
        * delta_h
        * distance
        / _C_KM_S
    )
    if family == "zero-noise":
        offset, sigma = 0.0, math.sqrt(float(config["scoring"]["zero_noise_variance"]))
    elif family == "distance-measurement":
        offset, sigma = distance_draw, max(minimum, sigma_distance)
    elif family == "bulk-shear-flow":
        offset, sigma = flow, max(minimum, 250.0 / _C_KM_S)
    elif family == "distance-plus-flow":
        offset = distance_draw + flow
        sigma = max(minimum, math.hypot(sigma_distance, 250.0 / _C_KM_S))
    elif family == "selection-mask-boundary":
        offset, sigma = boundary, max(minimum, abs(boundary) + minimum)
    else:
        raise SchemaViolation("unknown noise family")
    response = np.asarray(prediction + offset, dtype=np.float64)
    variance = np.asarray([sigma * sigma], dtype=np.float64)
    _require(np.all(np.isfinite(response)) and np.all(variance > 0.0), "invalid synthetic response")
    return (
        response,
        variance,
        {
            "family": family,
            "offset_hex": offset.hex(),
            "sigma_hex": sigma.hex(),
            "derived_seed": str(lineage.derived_seed),
        },
    )


def _feature_unit(feature: str) -> str:
    if feature.endswith("-mpc"):
        return "Mpc"
    if "km-s-mpc" in feature:
        return "km s^-1 Mpc^-1"
    if "modulus" in feature:
        return "mag"
    return "1"


def _feature_axes(feature: str) -> tuple[str, ...]:
    if feature == "source.vector.direction-cartesian":
        return ("cartesian",)
    if feature == "source.vector.flow-shear-design":
        return ("nuisance",)
    return ("object",)


def _scenario(
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    scenario_id: str,
    truth_world_id: str,
    nuisance_draw: int,
    response: np.ndarray,
    variance: np.ndarray,
    truth_index: int,
) -> ScenarioDescriptor:
    anchors = tuple(
        AnchorBinding(row["id"].lower(), row["path"], row["sha256"])
        for row in sorted(
            (*config["source_anchors"], *config["contract_bindings"]),
            key=lambda row: row["id"].lower(),
        )
    )
    truth = np.asarray([truth_index], dtype=np.int64)
    values = item["values"]
    lineage = SeedLineage(
        int(config["suite_seed"]),
        scenario_id,
        f"cf4-{item['identifier']}",
        truth_world_id,
        nuisance_draw,
        0,
    )
    return ScenarioDescriptor(
        scenario_id=scenario_id,
        object_id=f"cf4-{item['identifier']}",
        experiment_id=config["experiment_id"],
        domain="void-cosmology",
        geometry_mode=config["geometry_mode"],
        time_mode=config["time_mode"],
        coordinate_frame=config["coordinate_frame"],
        axes=(
            AxisSpec("cartesian", 3, None, None),
            AxisSpec("nuisance", 9, None, None),
            AxisSpec("object", 1, None, None),
        ),
        formula_features=tuple(
            FeatureValueRef(
                feature,
                VALUES_PATH.as_posix(),
                array_sha256(values[feature]),
                values[feature].dtype.name,
                values[feature].shape,
                _feature_axes(feature),
                _feature_unit(feature),
                config["coordinate_frame"],
            )
            for feature in _FEATURES
        ),
        scoring_responses=(
            FeatureValueRef(
                _RESPONSE,
                VALUES_PATH.as_posix(),
                array_sha256(response),
                "float64",
                (1,),
                ("object",),
                "1",
                config["coordinate_frame"],
            ),
        ),
        hidden_truth=(
            FeatureValueRef(
                _TRUTH,
                VALUES_PATH.as_posix(),
                array_sha256(truth),
                "int64",
                (1,),
                ("object",),
                "integer code",
                "latent",
            ),
        ),
        expected_predictions=(
            EmittedPredictionSpec(
                _OUTPUT,
                VALUES_PATH.as_posix(),
                "float64",
                (1,),
                ("object",),
                "1",
                config["coordinate_frame"],
            ),
        ),
        uncertainties=(
            UncertaintyRef(
                _UNCERTAINTY,
                _RESPONSE,
                "diagonal-covariance",
                VALUES_PATH.as_posix(),
                array_sha256(variance),
            ),
        ),
        anchors=anchors,
        seed_lineage=lineage,
    )


def _array_key(*parts: str) -> str:
    return "__".join(
        "".join(character.lower() if character.isalnum() else "-" for character in part).strip("-")
        for part in parts
    )


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            value_buffer = io.BytesIO()
            np.lib.format.write_array(value_buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, value_buffer.getvalue())
    return buffer.getvalue()


def derive_release() -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, bytes]:
    config = load_config()
    validate_config(config)
    catalogue = _catalogue(config)
    bindings = _bindings(config)
    validate_binding_catalogue(bindings, catalogue)
    registrations = tuple(
        AdapterRegistration.create(f"adapter.void-cosmology.{row.formula_id.lower()}.v1", row)
        for row in bindings
        if row.status is BindingStatus.EXECUTABLE
    )
    validate_adapter_registry(registrations)
    geometry = _parse_vast_geometry(config)
    source_rows, source_access = _select_source_objects(config, geometry)
    items = _variant_items(source_rows, config)
    arrays: dict[str, np.ndarray] = {}
    scenarios: list[ScenarioDescriptor] = []
    scenario_values: dict[str, ScenarioRuntimeValues] = {}
    truths: dict[str, str] = {}
    comparisons: dict[str, tuple[ObservableComparison, ...]] = {}
    scenario_rows: list[dict[str, Any]] = []
    truth_ids = list(config["truth_formula_ids"])
    for item in items:
        for truth_index, truth_formula_id in enumerate(truth_ids):
            truth_prediction = _prediction(
                item["values"],
                {
                    "C01_OBSERVER_ENDPOINT_LOCAL_VOID": "source.scalar.observer-endpoint-chord-mpc",
                    "C02_TARGET_ENDPOINT_LOCAL_VOID": "source.scalar.target-endpoint-chord-mpc",
                    "C03_SINGLE_DOMINANT_VOID": "source.scalar.maximum-chord-mpc",
                    "C04_BOUNDED_FRACTION_NULL": "source.scalar.null-void-length-mpc",
                    "VQ00_STANDARD_FLRW_FLOW_CONTROL": None,
                    "VQ08_TWO_PHASE_VOID_FRACTION": "source.scalar.void-length-mpc",
                }[truth_formula_id],
            )[_OUTPUT]
            for nuisance_draw, family in enumerate(config["noise_families"]):
                scenario_id = (
                    f"void.cf4-{item['identifier']}.{item['variant']}."
                    f"truth-{truth_formula_id.lower()}.noise-{family}.v1"
                )
                truth_world_id = f"truth.{truth_formula_id.lower()}"
                lineage = SeedLineage(
                    int(config["suite_seed"]),
                    scenario_id,
                    f"cf4-{item['identifier']}",
                    truth_world_id,
                    nuisance_draw,
                    0,
                )
                response, variance, noise = _noise_response(
                    truth_prediction, item["values"], family, lineage, config
                )
                scenario = _scenario(
                    config,
                    item,
                    scenario_id,
                    truth_world_id,
                    nuisance_draw,
                    response,
                    variance,
                    truth_index,
                )
                truth_value = np.asarray([truth_index], dtype=np.int64)
                scenarios.append(scenario)
                scenario_values[scenario_id] = ScenarioRuntimeValues(
                    formula_values=item["values"],
                    response_values={_RESPONSE: response},
                    truth_values={_TRUTH: truth_value},
                    uncertainty_values={_UNCERTAINTY: variance},
                )
                truths[scenario_id] = truth_formula_id
                comparisons[scenario_id] = (ObservableComparison(_OUTPUT, _RESPONSE, _UNCERTAINTY),)
                locators: dict[str, dict[str, str]] = {}
                for feature, value in item["values"].items():
                    key = _array_key(
                        "feature", str(item["identifier"]), str(item["variant"]), feature
                    )
                    arrays[key] = value
                    locators[feature] = {"key": key, "sha256": array_sha256(value)}
                for label, value in (
                    ("response", response),
                    ("variance", variance),
                    ("truth", truth_value),
                ):
                    key = _array_key(label, scenario_id)
                    arrays[key] = value
                    locators[label] = {"key": key, "sha256": array_sha256(value)}
                scenario_rows.append(
                    {
                        "scenario": scenario.to_dict(),
                        "scenario_sha256": scenario.content_sha256,
                        "truth_formula_id": truth_formula_id,
                        "geometry_variant": item["variant"],
                        "source_geometry": item["source_geometry"],
                        "source_index": item["source_index"],
                        "noise": noise,
                        "geometry_values": {
                            key: item[key]
                            for key in (
                                "distance_mpc_hex",
                                "void_length_mpc_hex",
                                "null_void_length_mpc_hex",
                                "void_fraction_hex",
                            )
                        },
                        "value_locators": locators,
                    }
                )
    scenarios = sorted(scenarios, key=lambda row: row.scenario_id)
    scenario_rows = sorted(scenario_rows, key=lambda row: row["scenario"]["scenario_id"])
    release = SyntheticSuiteRelease(
        suite_id="gravity.synthetic.void-cosmology-source-shaped-matrix.v1",
        version="v1.0.0",
        release_sha256=canonical_sha256(
            {
                "config": _file_sha256(_ROOT / CONFIG_PATH),
                "scenario_ids": [row.scenario_id for row in scenarios],
            }
        ),
        ontology_sha256=catalogue.content_sha256,
        generator_sha256=_file_sha256(Path(__file__)),
        observation_operator_sha256=_json_sha256(config["noise"]),
        changed_feature_ids=tuple(sorted((*_FEATURES, _OUTPUT, _RESPONSE, _TRUTH))),
        change_level="MAJOR",
        response_calibrated=False,
        prediction_semantics_changed=True,
    )
    parameter_cells = {
        row.binding_id: (
            (ParameterCell("fixed-source-contract", {}),)
            if row.status is BindingStatus.EXECUTABLE
            else ()
        )
        for row in bindings
    }
    result = run_discovery_matrix_v2(
        catalogue=catalogue,
        release=release,
        scenarios=scenarios,
        scenario_values=scenario_values,
        truth_formula_by_scenario=truths,
        bindings=bindings,
        adapters=registrations,
        parameter_cells=parameter_cells,
        comparisons=comparisons,
        distinct_gap=float(config["scoring"]["distinct_gap"]),
        ledger_id="gravity.synthetic.void-cosmology-source-shaped-matrix.v1.ledger",
    )
    cells_by_scenario: dict[str, list[Any]] = {}
    for cell in result.cells:
        cells_by_scenario.setdefault(cell.scenario_id, []).append(cell)
    confusion = {truth: {candidate: 0 for candidate in sorted(_EXECUTABLE)} for truth in truth_ids}
    recovery = {truth: {"scenarios": 0, "recovered": 0, "distinct": 0} for truth in truth_ids}
    for scenario_id in sorted(cells_by_scenario):
        cells = cells_by_scenario[scenario_id]
        truth = truths[scenario_id]
        winners = [cell.formula_id for cell in cells if cell.winner]
        for winner in winners:
            confusion[truth][winner] += 1
        recovery[truth]["scenarios"] += 1
        recovery[truth]["recovered"] += int(any(cell.truth_recovered for cell in cells))
        recovery[truth]["distinct"] += int(
            any(cell.truth_recovered and cell.distinct for cell in cells)
        )
    values_bytes = _npz_bytes(arrays)
    _require(values_bytes == _npz_bytes(arrays), "NPZ serialization nondeterministic")
    scenarios_bytes = b"".join(_json_bytes(row) + b"\n" for row in scenario_rows)
    ledger_bytes = _json_bytes(result.ledger.to_dict(), indent=2)
    confusion_payload = {
        "schema": "open-gravity-void-cosmology-confusion-matrix-1.0",
        "truth_formula_ids": truth_ids,
        "candidate_formula_ids": sorted(_EXECUTABLE),
        "winner_membership_counts": confusion,
        "recovery_by_truth": recovery,
        "scenario_count": result.scenario_count,
        "attempted_cell_count": result.attempted_cell_count,
        "scored_cell_count": result.scored_cell_count,
        "truth_recovery_count": result.truth_recovery_count,
        "distinct_truth_recovery_count": result.distinct_truth_recovery_count,
        "runner_result_content_sha256": result.content_sha256,
        "no_hand_ranking": True,
    }
    confusion_bytes = _json_bytes(confusion_payload, indent=2)
    all_lengths = [float(item["values"]["source.scalar.void-length-mpc"][0]) for item in items]
    all_distances = [float(item["values"]["source.scalar.distance-mpc"][0]) for item in items]
    all_nulls = [float(item["values"]["source.scalar.null-void-length-mpc"][0]) for item in items]
    geometry_gate = all(
        0.0 <= length <= distance
        for length, distance in zip(all_lengths, all_distances, strict=True)
    )
    null_gate = all(
        0.0 <= length <= distance for length, distance in zip(all_nulls, all_distances, strict=True)
    )
    diagnostics = {
        "schema": "open-gravity-void-cosmology-geometry-identifiability-1.0",
        "geometry_valid": geometry_gate,
        "bounded_fraction_null_valid": null_gate,
        "absolute_length_permutation_used": False,
        "permutation_rule": "permute dimensionless void fraction within four-object distance strata; reconstruct L'=f_perm*D for each target",
        "object_count": len(source_rows),
        "source_definition_count": len(config["geometry_variants"]),
        "variant_item_count": len(items),
        "vast_source_counts": {
            "table1_rows": geometry["table1_rows"],
            "table2_rows": geometry["table2_rows"],
            "retained_spheres": geometry["retained"],
        },
        "selected_cf4": [
            {
                "identifier": str(row["identifier"]),
                "source_index": int(row["source_index"]),
                "bucket": int(row["bucket"]),
                "role": row["role"],
                "distance_path_mpc_hex": float(row["distance_path_mpc"]).hex(),
                "mask_neighborhood_fraction_hex": float(row["mask_neighborhood_fraction"]).hex(),
            }
            for row in source_rows
        ],
        "blocked_formula_ids": sorted(_BLOCKED),
        "blocked_reasons": config["adapter_blocks"],
        "claim_class": config["claim_class"],
    }
    _require(geometry_gate and null_gate, "geometry-valid null invariant failed")
    diagnostics_bytes = _json_bytes(diagnostics, indent=2)
    receipt_body = {
        "schema": "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-receipt-1.0",
        "package_id": config["package_id"],
        "version": config["version"],
        "status": "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT",
        "claim_class": config["claim_class"],
        "scientific_claim": "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION",
        "independent_audit_completed": False,
        "distinct_independent_audit_required": True,
        "object_count": len(source_rows),
        "geometry_variant_count": len(config["geometry_variants"]),
        "noise_family_count": len(config["noise_families"]),
        "truth_formula_count": len(truth_ids),
        "scenario_count": result.scenario_count,
        "attempted_cell_count": result.attempted_cell_count,
        "scored_cell_count": result.scored_cell_count,
        "replay_entry_count": len(result.ledger.entries),
        "truth_recovery_count": result.truth_recovery_count,
        "distinct_truth_recovery_count": result.distinct_truth_recovery_count,
        "recovery_by_truth": recovery,
        "executable_formula_ids": sorted(_EXECUTABLE),
        "blocked_formula_ids": sorted(_BLOCKED),
        "formula_bindings": {row.formula_id: row.to_dict() for row in bindings},
        "formula_binding_sha256": {row.formula_id: row.content_sha256 for row in bindings},
        "adapter_sha256": {
            row.formula_binding.formula_id: row.adapter_sha256 for row in registrations
        },
        "source_anchor_sha256": {row["id"]: row["sha256"] for row in config["source_anchors"]},
        "contract_binding_sha256": {
            row["id"]: row["sha256"] for row in config["contract_bindings"]
        },
        "geometry_gates": {
            "zero_le_l_void_le_distance": geometry_gate,
            "zero_le_l_null_le_distance": null_gate,
            "absolute_length_permutation_used": False,
        },
        "package_hashes": {
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
            "module_raw_sha256": _file_sha256(Path(__file__)),
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH),
        },
        "artifact_sha256": {
            "values.npz": hashlib.sha256(values_bytes).hexdigest(),
            "scenarios.jsonl": hashlib.sha256(scenarios_bytes).hexdigest(),
            "ledger.json": hashlib.sha256(ledger_bytes).hexdigest(),
            "confusion-matrix.json": hashlib.sha256(confusion_bytes).hexdigest(),
            "geometry-and-identifiability.json": hashlib.sha256(diagnostics_bytes).hexdigest(),
        },
        "access_accounting": {
            **config["access_contract"],
            **source_access,
            "vast1_source_rows_decoded": geometry["table1_rows"],
            "vast2_source_rows_decoded": geometry["table2_rows"],
            "synthetic_response_values_generated": result.scenario_count,
            "real_scores": 0,
        },
        "limitations": [
            "All response vectors are generated from frozen source geometry and synthetic nuisance draws; no empirical response is scored.",
            "The CF4 gzip row container is traversed only after exact ledger verification; only 1PGC, DMzp, e_DMzp, Dist, RAdeg, and DEdeg are decoded.",
            "V3k, published peculiar velocities, measured redshift residuals, validation/confirmation source fields, and Pantheon remain unopened and undecoded.",
            "VQ01-VQ07 and VQ09-VQ10 are retained as SOURCE_BLOCKED; a void mask is not substituted for their missing Q, rho_b, J_g, or reservoir state.",
            "The bounded null permutes dimensionless exposure fractions within distance strata and reconstructs each length from that target's own distance.",
        ],
    }
    receipt = {**receipt_body, "content_sha256": _json_sha256(receipt_body)}
    return receipt, values_bytes, scenarios_bytes, ledger_bytes, confusion_bytes, diagnostics_bytes


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing changed artifact: {path}")
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
            _require(path.read_bytes() == payload, f"concurrent changed artifact: {path}")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def freeze() -> str:
    receipt, values, scenarios, ledger, confusion, diagnostics = derive_release()
    payloads = (
        (VALUES_PATH, values),
        (SCENARIOS_PATH, scenarios),
        (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion),
        (DIAGNOSTICS_PATH, diagnostics),
        (RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    return ":".join(_write_once(_ROOT / path, payload) for path, payload in payloads)


def check() -> dict[str, Any]:
    receipt, values, scenarios, ledger, confusion, diagnostics = derive_release()
    payloads = (
        (VALUES_PATH, values),
        (SCENARIOS_PATH, scenarios),
        (LEDGER_PATH, ledger),
        (CONFUSION_PATH, confusion),
        (DIAGNOSTICS_PATH, diagnostics),
        (RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    for path, payload in payloads:
        _require(
            (_ROOT / path).is_file() and (_ROOT / path).read_bytes() == payload,
            f"stored artifact differs: {path}",
        )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze())
    else:
        print(json.dumps(check(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
