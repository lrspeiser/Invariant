"""Response-blind X-COP real-source-shaped synthetic injection matrix.

Only the eight frozen density profiles and five declared stellar profiles are
opened.  Spherical source profiles are lifted onto the already frozen 17^3
Lane-6 adapter grid.  Hidden responses and noise are synthetic.  Pressure,
temperature, hydrostatic mass, lensing, and motion responses are never opened.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    gravity_gain_persistence_gp01_xcop_source_preflight as source_preflight,
)
from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base
from sigma_theory_compiler import (
    open_gravity_full3d_phangs_synthetic_adapter_preflight_v1 as frozen_adapters,
)
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
    EligibilityStatus,
    FormulaExecutionBinding,
    ResourceBounds,
    validate_binding_catalogue,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    SyntheticReplayLedger,
    SyntheticSuiteRelease,
    status_from_result,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import (
    AnchorBinding,
    AxisSpec,
    EmittedPredictionSpec,
    FeatureValueRef,
    ScenarioDescriptor,
    UncertaintyRef,
    array_sha256,
    decide_scenario_eligibility,
    execute_binding_in_process,
    validate_scenario_catalogue,
    validate_scenario_values,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

CONFIG_PATH = Path(
    "configs/open_gravity_xcop_real_source_shaped_synthetic_injection_matrix_v1.json"
)
TEST_PATH = Path("tests/test_open_gravity_xcop_real_source_shaped_synthetic_injection_matrix_v1.py")
OUTPUT_DIR = Path("runs/gravity/open-gravity-xcop-real-source-shaped-synthetic-injection-matrix-v1")
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]
_A0 = 1.2e-10
_FORMULA_FEATURES = frozen_adapters._FORMULA_FEATURES
_EXECUTABLE_ENTRYPOINTS = {
    key: value
    for key, value in frozen_adapters._MECHANISM_ENTRYPOINTS.items()
    if key != "DPEL01_DISK_POLAR_ESCAPE_LOAD"
}
_DPEL = "DPEL01_DISK_POLAR_ESCAPE_LOAD"


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
        raise SchemaViolation("X-COP synthetic path escaped repository")
    path = (_ROOT / parsed.as_posix()).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("X-COP synthetic path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_upstreams: bool = True) -> None:
    expected = {
        "schema",
        "package_id",
        "version",
        "status",
        "claim_class",
        "experiment_id",
        "suite_seed",
        "clusters",
        "mechanisms",
        "noise_families",
        "grid",
        "noise",
        "scoring",
        "parameter_schema_path",
        "output_directory",
        "upstream_bindings",
        "access_contract",
        "adapter_blocks",
    }
    _require(set(config) == expected, "X-COP synthetic config keys changed")
    _require(
        config["schema"] == "open-gravity-xcop-real-source-shaped-synthetic-injection-matrix-1.0",
        "X-COP synthetic schema changed",
    )
    _require(
        config["package_id"] == "open-gravity-xcop-real-source-shaped-synthetic-injection-matrix-v1"
        and config["version"] == "v1.0.0",
        "X-COP synthetic package identity changed",
    )
    _require(config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE", "status changed")
    _require(config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL", "claim changed")
    _require(
        config["clusters"]
        == ["A1644", "A1795", "A2142", "A2255", "A2319", "A3266", "A85", "ZW1215"],
        "X-COP cluster set changed",
    )
    _require(config["mechanisms"] == sorted(_EXECUTABLE_ENTRYPOINTS), "mechanisms changed")
    _require(
        config["noise_families"]
        == ["independent-source-envelope", "radial-correlated-source-envelope", "zero-noise"],
        "noise families changed",
    )
    grid = config["grid"]
    _require(
        grid["points_per_axis"] == 17
        and grid["half_box_over_outer_source_radius"] == 1.25
        and grid["geometry_mode"] == "spherical-lifted3d"
        and grid["time_mode"] == "static",
        "spherical lift contract changed",
    )
    noise = config["noise"]
    _require(
        noise
        == {
            "fraction_floor": 0.01,
            "fraction_ceiling": 0.5,
            "response_floor_fraction_of_rms": 0.001,
            "radial_control_knots": 9,
        },
        "noise contract changed",
    )
    scoring = config["scoring"]
    _require(
        scoring["primary_metric"] == "whitened_rmse"
        and scoring["secondary_metric"] == "relative_rmse"
        and scoring["winner_absolute_tolerance"] == 1.0e-12
        and scoring["minimum_whitened_gap_for_distinct_signature"] == 0.1
        and scoring["no_hand_ranking"] is True,
        "calculated scoring contract changed",
    )
    _require(
        _repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(), "output changed"
    )
    access = config["access_contract"]
    _require(access["allowed_source_roles"] == ["density", "stellar_mass"], "source roles changed")
    _require(
        {"pressure", "temperature", "lensing", "hydrostatic_mass", "inferred_total_mass"}
        <= set(access["forbidden_roles"]),
        "response exclusion changed",
    )
    _require(
        access["expected_source_files_opened"] == 13
        and access["expected_source_bytes_opened"] == 308160
        and all(
            access[key] == 0
            for key in (
                "measured_response_files_opened",
                "measured_response_rows_opened",
                "network_calls",
                "model_calls",
                "paid_calls",
            )
        ),
        "access ceiling changed",
    )
    _require(len(config["adapter_blocks"]) == 4, "adapter block inventory changed")
    _require(config["adapter_blocks"][0]["formula_id"] == _DPEL, "DPEL block changed")
    upstream_ids = [row["id"] for row in config["upstream_bindings"]]
    _require(upstream_ids == sorted(set(upstream_ids)), "upstream bindings must be sorted")
    for row in config["upstream_bindings"]:
        _require(set(row) == {"id", "path", "sha256"}, "upstream binding schema changed")
        if verify_upstreams:
            path = _repo_path(row["path"])
            _require(
                path.is_file() and _file_sha256(path) == row["sha256"],
                f"upstream changed: {row['id']}",
            )


def _upstream_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return {row["id"]: _file_sha256(_repo_path(row["path"])) for row in config["upstream_bindings"]}


def _catalogue(config: Mapping[str, Any]):
    provenance = canonical_sha256(config["upstream_bindings"])
    specs = (
        ("geometry.scalar.grid-spacing-normalized", "normalized grid spacing", 0, "1", ("object",)),
        ("geometry.scalar.half-box-length", "solver half-box length", 0, "m", ("object",)),
        ("geometry.scalar.x-coordinate", "x coordinate", 0, "m", ("x",)),
        ("geometry.scalar.y-coordinate", "y coordinate", 0, "m", ("y",)),
        ("geometry.scalar.z-coordinate", "z coordinate", 0, "m", ("z",)),
        ("geometry.vector.disk-normal", "unused compatibility axis", 1, "1", ("component",)),
        (
            "source.scalar.expected-mass-normalized",
            "normalized enclosed baryonic mass",
            0,
            "1",
            ("object",),
        ),
        (
            "source.scalar.mass-density",
            "spherical-lifted baryonic density",
            0,
            "kg m^-3",
            ("x", "y", "z"),
        ),
        (
            "source.scalar.solver-density",
            "Lane6 normalized baryonic density",
            0,
            "1",
            ("x", "y", "z"),
        ),
        (
            "response.vector.synthetic-acceleration",
            "hidden synthetic acceleration",
            1,
            "m s^-2",
            ("x", "y", "z", "component"),
        ),
        (
            "prediction.vector.acceleration",
            "candidate acceleration",
            1,
            "m s^-2",
            ("x", "y", "z", "component"),
        ),
        (
            "truth.scalar.injection-id",
            "synthetic mechanism identity",
            0,
            "typed hidden value",
            ("object",),
        ),
    )
    dimensions = {
        "1": (0, 0, 0, 0, 0, 0, 0),
        "typed hidden value": (0, 0, 0, 0, 0, 0, 0),
        "m": (0, 1, 0, 0, 0, 0, 0),
        "kg m^-3": (1, -3, 0, 0, 0, 0, 0),
        "m s^-2": (0, 1, -2, 0, 0, 0, 0),
    }
    elements = []
    for element_id, quantity, rank, unit, axes in specs:
        if element_id.startswith("response."):
            role, availability = DataRole.SCORING_ONLY_RESPONSE, Availability.SYNTHETIC_ONLY
        elif element_id.startswith("prediction."):
            role, availability = DataRole.DERIVED, Availability.SYNTHETIC_ONLY
        elif element_id.startswith("truth."):
            role, availability = DataRole.LATENT_SYNTHETIC_TRUTH, Availability.SYNTHETIC_ONLY
        else:
            role, availability = DataRole.FORMULA_INPUT, Availability.PUBLIC_SOURCE
        elements.append(
            DataElement(
                element_id=element_id,
                namespace=element_id.rsplit(".", 1)[0],
                physical_quantity=quantity,
                tensor_rank=rank,
                si_dimension=dimensions[unit],
                canonical_unit=unit,
                frame="latent" if element_id.startswith("truth.") else "solver-source",
                support="17^3 spherical lift of frozen X-COP radial baryonic source profiles",
                axes=axes,
                component="total",
                derivation_parents=(),
                uncertainty=(
                    UncertaintyKind.COVARIANCE
                    if element_id.startswith(("response.", "prediction."))
                    else UncertaintyKind.NONE
                ),
                availability=availability,
                experiment_roles=(ExperimentRole(config["experiment_id"], role),),
                provenance_sha256=provenance,
            )
        )
    return catalogue_from_elements("open-gravity-xcop-source-shaped-synthetic", "v1.0.0", elements)


def _bindings(config: Mapping[str, Any]) -> tuple[FormulaExecutionBinding, ...]:
    upstream = _upstream_hashes(config)
    schema_sha = _file_sha256(_repo_path(config["parameter_schema_path"]))
    rows = []
    all_ids = sorted((*config["mechanisms"], _DPEL))
    for mechanism in all_ids:
        executable = mechanism != _DPEL
        entrypoint = None
        if executable:
            entrypoint = (
                "sigma_theory_compiler.open_gravity_full3d_phangs_synthetic_adapter_preflight_v1:"
                + _EXECUTABLE_ENTRYPOINTS[mechanism]
            )
        formula_sha = canonical_sha256(
            {
                "mechanism_id": mechanism,
                "frozen_adapter_module_sha256": upstream["FULL3D_ADAPTERS"],
                "frozen_solver_module_sha256": upstream["LANE6_FROZEN_SOLVERS"],
                "spherical_source_lift": "17^3-exact-outer-mass-normalized-v1",
            }
        )
        rows.append(
            FormulaExecutionBinding(
                binding_id=f"binding.cluster.{mechanism.lower()}.v1",
                formula_id=mechanism,
                formula_version="v1.0.0-frozen-source-lift-adapter",
                formula_sha256=formula_sha,
                status=BindingStatus.EXECUTABLE if executable else BindingStatus.UNADAPTED,
                entrypoint=entrypoint,
                required_features=tuple(_FORMULA_FEATURES),
                optional_features=(),
                emitted_features=("prediction.vector.acceleration",),
                domains=("cluster",),
                geometry_support=(("spherical-lifted3d",) if executable else ("disk3d",)),
                time_support=("static",),
                parameter_schema_path=config["parameter_schema_path"],
                parameter_schema_sha256=schema_sha,
                approximation_ceiling=(
                    "disk-only adapter blocked on spherical X-COP geometry"
                    if not executable
                    else "synthetic-only 17^3 spherical lift of radial public source profiles"
                ),
                health_gates=("determinism", "finite-output", "source-hash", "typed-output"),
                resource_bounds=ResourceBounds(300, 2_000_000_000, 2_000_000),
            )
        )
    return tuple(rows)


def _source_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    row = next(
        item
        for item in config["upstream_bindings"]
        if item["id"] == "XCOP_SOURCE_PREFLIGHT_RECEIPT"
    )
    receipt = json.loads(_repo_path(row["path"]).read_text(encoding="utf-8"))
    _require(
        receipt.get("content_sha256") == source_preflight.receipt_content_sha256(receipt),
        "source receipt content hash failed",
    )
    _require(
        receipt["status"] == "SOURCE_ONLY_PREFLIGHT_COMPLETE_ZERO_RESPONSE_ACCESS",
        "source receipt status changed",
    )
    access = receipt["source_access"]
    _require(
        access["source_files_opened"] == 13
        and access["pressure_files_opened"] == 0
        and access["temperature_files_opened"] == 0
        and access["response_rows_opened"] == 0,
        "source receipt response boundary changed",
    )
    return receipt


def _source_items(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_config = source_preflight.load_config(_ROOT)
    raw_root, records, item59 = source_preflight._source_records(_ROOT, source_config)
    record_by_key = {(row["cluster"], row["role"]): row for row in records}
    constants = item59["constants"]
    grid = base.make_grid(int(config["grid"]["points_per_axis"]))
    results: list[dict[str, Any]] = []
    opened: list[dict[str, Any]] = []
    for cluster in config["clusters"]:

        def read_role(role: str, cluster_key: str = cluster) -> bytes:
            record = record_by_key[(cluster_key, role)]
            path = raw_root / record["member"]
            payload = path.read_bytes()
            _require(len(payload) == record["bytes"], "source byte count changed")
            _require(hashlib.sha256(payload).hexdigest() == record["sha256"], "source hash changed")
            opened.append(dict(record))
            return payload

        density, density_header = source_preflight._fits_table(
            read_role("density"), source_config["source_contract"]["density_hdu"]
        )
        _require(
            list(density.dtype.names or ()) == source_config["source_contract"]["density_columns"],
            "density schema changed",
        )
        radius_kpc = np.asarray(density["RW_X"], dtype=np.float64) * float(density_header["R500"])
        ne = np.maximum(np.asarray(density["NE"], dtype=np.float64), np.finfo(np.float64).tiny)
        ne_lo = np.maximum(np.asarray(density["ERR_NE_LO"], dtype=np.float64), 0.0)
        ne_hi = np.maximum(np.asarray(density["ERR_NE_HI"], dtype=np.float64), 0.0)
        _require(
            radius_kpc.size >= 3 and np.all(np.diff(radius_kpc) > 0.0), "density radius invalid"
        )
        radius_m = radius_kpc * float(constants["kiloparsec_m"])
        gas_density = (
            ne
            * 1.0e6
            * float(constants["mean_molecular_weight_per_electron"])
            * float(constants["proton_mass_kg"])
        )
        gas_mass = source_preflight._cumulative_mass(radius_m, gas_density)
        stellar = None
        stellar_fraction = 0.0
        if (cluster, "stellar_mass") in record_by_key:
            table, _ = source_preflight._fits_table(
                read_role("stellar_mass"), source_config["source_contract"]["stellar_hdu"]
            )
            _require(
                list(table.dtype.names or ())
                == source_config["source_contract"]["stellar_columns"],
                "stellar schema changed",
            )
            stellar = {
                "radius_kpc": np.asarray(table["RADIUS"], dtype=np.float64),
                "mass_msun": np.asarray(table["MSTAR"], dtype=np.float64),
                "mass_low_msun": np.asarray(table["MSTAR_LO"], dtype=np.float64),
                "mass_high_msun": np.asarray(table["MSTAR_HI"], dtype=np.float64),
            }
            denom = np.maximum(np.abs(stellar["mass_msun"]), np.finfo(np.float64).tiny)
            stellar_fraction = float(
                np.median(
                    np.abs(stellar["mass_high_msun"] - stellar["mass_low_msun"]) / (2.0 * denom)
                )
            )
        variant = {
            "nuisances": {
                "published_stellar_mass_scale": source_config["frozen_source_mapping"][
                    "published_stellar_mass_scale"
                ],
                "missing_stellar_to_gas_mass_ratio": source_config["frozen_source_mapping"][
                    "missing_stellar_to_gas_mass_ratio"
                ],
            }
        }
        stellar_mass = source_preflight._member_mass(
            {"stellar": stellar}, radius_kpc, gas_mass, variant, "nominal", item59
        )
        baryonic_mass = gas_mass + stellar_mass
        effective_density = source_preflight._effective_density(radius_m, baryonic_mass)
        outer_mass = float(baryonic_mass[-1])
        outer_radius_m = float(radius_m[-1])
        half_box_m = outer_radius_m * float(config["grid"]["half_box_over_outer_source_radius"])
        x, y, z = np.meshgrid(grid.coordinates, grid.coordinates, grid.coordinates, indexing="ij")
        physical_radius = np.sqrt(x * x + y * y + z * z) * half_box_m
        lifted = np.interp(
            np.minimum(physical_radius, outer_radius_m),
            radius_m,
            effective_density,
            left=float(effective_density[0]),
            right=float(effective_density[-1]),
        )
        lifted = np.where(physical_radius <= outer_radius_m, lifted, 0.0).astype(np.float64)
        cell_volume = (grid.spacing * half_box_m) ** 3
        raw_lift_mass = float(np.sum(lifted) * cell_volume)
        _require(raw_lift_mass > 0.0 and outer_mass > 0.0, "spherical lift mass invalid")
        mass_scale = outer_mass / raw_lift_mass
        lifted *= mass_scale
        gravity = float(constants["gravity_si"])
        solver_density = lifted * gravity * half_box_m / _A0
        expected_mass = gravity * outer_mass / (_A0 * half_box_m**2)
        coordinates = (grid.coordinates * half_box_m).astype(np.float64)
        values = {
            "geometry.scalar.grid-spacing-normalized": np.asarray([grid.spacing], dtype=np.float64),
            "geometry.scalar.half-box-length": np.asarray([half_box_m], dtype=np.float64),
            "geometry.scalar.x-coordinate": coordinates,
            "geometry.scalar.y-coordinate": coordinates,
            "geometry.scalar.z-coordinate": coordinates,
            "geometry.vector.disk-normal": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "source.scalar.expected-mass-normalized": np.asarray([expected_mass], dtype=np.float64),
            "source.scalar.mass-density": lifted.astype(np.float64),
            "source.scalar.solver-density": solver_density.astype(np.float64),
        }
        density_fraction = float(np.median((ne_lo + ne_hi) / (2.0 * ne)))
        stellar_mass_fraction = float(stellar_mass[-1] / outer_mass)
        missing_component = 0.1 if stellar is None else 0.0
        combined = math.sqrt(
            density_fraction**2
            + (stellar_mass_fraction * stellar_fraction) ** 2
            + missing_component**2
        )
        fractional_sigma = float(
            np.clip(
                combined, config["noise"]["fraction_floor"], config["noise"]["fraction_ceiling"]
            )
        )
        results.append(
            {
                "object_id": cluster,
                "values": values,
                "grid_radius_normalized": np.sqrt(x * x + y * y + z * z).astype(np.float64),
                "source_metadata": {
                    "density_rows": int(radius_kpc.size),
                    "stellar_rows": 0 if stellar is None else int(stellar["radius_kpc"].size),
                    "stellar_profile_available": stellar is not None,
                    "missing_stellar_rule_applied": stellar is None,
                    "outer_source_radius_kpc": float(radius_kpc[-1]),
                    "outer_baryonic_mass_kg": outer_mass,
                    "half_box_m": half_box_m,
                    "raw_lift_mass_kg": raw_lift_mass,
                    "lift_mass_scale": mass_scale,
                    "normalized_expected_mass": expected_mass,
                    "density_fractional_envelope_median": density_fraction,
                    "stellar_fractional_envelope_median": stellar_fraction,
                    "stellar_outer_baryonic_mass_fraction": stellar_mass_fraction,
                    "combined_fractional_sigma_unclipped": combined,
                    "combined_fractional_sigma": fractional_sigma,
                    "source_feature_hashes": {
                        key: array_sha256(value) for key, value in values.items()
                    },
                },
            }
        )
    _require(
        len(opened) == 13 and sum(row["bytes"] for row in opened) == 308160,
        "source access count changed",
    )
    _require(
        all(row["role"] in {"density", "stellar_mass"} for row in opened), "response source opened"
    )
    return results, sorted(opened, key=lambda row: (row["cluster"], row["role"])), source_config


def _scenario(
    *,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    scenario_id: str,
    truth_world_id: str,
    nuisance_draw: int,
    response: np.ndarray,
    variance: np.ndarray,
    truth_index: int,
) -> ScenarioDescriptor:
    values = item["values"]
    object_id = str(item["object_id"]).lower()
    units = {
        "geometry.scalar.grid-spacing-normalized": "1",
        "geometry.scalar.half-box-length": "m",
        "geometry.scalar.x-coordinate": "m",
        "geometry.scalar.y-coordinate": "m",
        "geometry.scalar.z-coordinate": "m",
        "geometry.vector.disk-normal": "1",
        "source.scalar.expected-mass-normalized": "1",
        "source.scalar.mass-density": "kg m^-3",
        "source.scalar.solver-density": "1",
    }
    axes = {
        "geometry.scalar.grid-spacing-normalized": ("object",),
        "geometry.scalar.half-box-length": ("object",),
        "geometry.scalar.x-coordinate": ("x",),
        "geometry.scalar.y-coordinate": ("y",),
        "geometry.scalar.z-coordinate": ("z",),
        "geometry.vector.disk-normal": ("component",),
        "source.scalar.expected-mass-normalized": ("object",),
        "source.scalar.mass-density": ("x", "y", "z"),
        "source.scalar.solver-density": ("x", "y", "z"),
    }
    truth = np.asarray([truth_index], dtype=np.int64)
    source_receipt_binding = next(
        row for row in config["upstream_bindings"] if row["id"] == "XCOP_SOURCE_PREFLIGHT_RECEIPT"
    )
    return ScenarioDescriptor(
        scenario_id=scenario_id,
        object_id=object_id,
        experiment_id=config["experiment_id"],
        domain="cluster",
        geometry_mode="spherical-lifted3d",
        time_mode="static",
        coordinate_frame="solver-source",
        axes=(
            AxisSpec("component", 3, None, None),
            AxisSpec("object", 1, None, None),
            AxisSpec(
                "x",
                17,
                "geometry.scalar.x-coordinate",
                array_sha256(values["geometry.scalar.x-coordinate"]),
            ),
            AxisSpec(
                "y",
                17,
                "geometry.scalar.y-coordinate",
                array_sha256(values["geometry.scalar.y-coordinate"]),
            ),
            AxisSpec(
                "z",
                17,
                "geometry.scalar.z-coordinate",
                array_sha256(values["geometry.scalar.z-coordinate"]),
            ),
        ),
        formula_features=tuple(
            FeatureValueRef(
                element_id,
                VALUES_PATH.as_posix(),
                array_sha256(values[element_id]),
                values[element_id].dtype.name,
                values[element_id].shape,
                axes[element_id],
                units[element_id],
                "solver-source",
            )
            for element_id in _FORMULA_FEATURES
        ),
        scoring_responses=(
            FeatureValueRef(
                "response.vector.synthetic-acceleration",
                VALUES_PATH.as_posix(),
                array_sha256(response),
                "float64",
                response.shape,
                ("x", "y", "z", "component"),
                "m s^-2",
                "solver-source",
            ),
        ),
        hidden_truth=(
            FeatureValueRef(
                "truth.scalar.injection-id",
                VALUES_PATH.as_posix(),
                array_sha256(truth),
                "int64",
                truth.shape,
                ("object",),
                "typed hidden value",
                "latent",
            ),
        ),
        expected_predictions=(
            EmittedPredictionSpec(
                "prediction.vector.acceleration",
                VALUES_PATH.as_posix(),
                "float64",
                response.shape,
                ("x", "y", "z", "component"),
                "m s^-2",
                "solver-source",
            ),
        ),
        uncertainties=(
            UncertaintyRef(
                "synthetic-acceleration.diagonal-covariance",
                "response.vector.synthetic-acceleration",
                "diagonal-covariance",
                VALUES_PATH.as_posix(),
                array_sha256(variance),
            ),
        ),
        anchors=(
            AnchorBinding(
                "xcop.source-only-preflight.v1",
                source_receipt_binding["path"],
                source_receipt_binding["sha256"],
            ),
        ),
        seed_lineage=SeedLineage(
            config["suite_seed"], scenario_id, object_id, truth_world_id, nuisance_draw, 0
        ),
    )


def _noise_response(
    truth: np.ndarray,
    family: str,
    lineage: SeedLineage,
    fractional_sigma: float,
    radius_normalized: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rms = max(float(np.sqrt(np.mean(truth * truth))), np.finfo(np.float64).tiny)
    floor = rms * float(config["noise"]["response_floor_fraction_of_rms"])
    sigma = fractional_sigma * np.maximum(np.abs(truth), floor)
    variance = np.square(sigma, dtype=np.float64)
    rng = np.random.default_rng(lineage.derived_seed)
    if family == "zero-noise":
        noise = np.zeros_like(truth)
    elif family == "independent-source-envelope":
        noise = rng.normal(size=truth.shape) * sigma
    else:
        _require(family == "radial-correlated-source-envelope", "unknown noise family")
        knot_count = int(config["noise"]["radial_control_knots"])
        knot_radius = np.linspace(0.0, float(np.max(radius_normalized)), knot_count)
        knot_draw = rng.normal(size=knot_count)
        radial_draw = np.interp(radius_normalized, knot_radius, knot_draw)
        noise = truth * (fractional_sigma * radial_draw[..., None])
    response = np.asarray(truth + noise, dtype=np.float64)
    _require(
        np.all(np.isfinite(response)) and np.all(np.isfinite(variance)) and np.all(variance > 0.0),
        "noise emitted invalid values",
    )
    return (
        response,
        variance,
        {
            "family": family,
            "derived_seed": lineage.derived_seed,
            "fractional_sigma": fractional_sigma,
            "noise_rms_m_s2": float(np.sqrt(np.mean(noise * noise))),
            "response_rms_m_s2": float(np.sqrt(np.mean(response * response))),
            "variance_representation": "diagonal marginal covariance; correlated family records radial draw separately",
        },
    )


def _metrics(candidate: np.ndarray, response: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    residual = candidate - response
    response_scale = max(float(np.sqrt(np.mean(response * response))), np.finfo(np.float64).tiny)
    return {
        "whitened_rmse": float(np.sqrt(np.mean(residual * residual / variance))),
        "relative_rmse": float(np.sqrt(np.mean(residual * residual)) / response_scale),
    }


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.asarray(arrays[key]) for key in sorted(arrays)})
    return buffer.getvalue()


def _array_key(*parts: str) -> str:
    return "__".join(part.lower().replace("-", "_").replace(".", "_") for part in parts)


def derive_release() -> tuple[dict[str, Any], bytes, bytes, bytes, bytes]:
    config = load_config()
    validate_config(config)
    source_receipt = _source_receipt(config)
    source_items, opened_sources, _source_config = _source_items(config)
    catalogue = _catalogue(config)
    bindings = _bindings(config)
    validate_binding_catalogue(bindings, catalogue)
    executable = tuple(row for row in bindings if row.status is BindingStatus.EXECUTABLE)
    registrations = tuple(
        AdapterRegistration.create(f"adapter.cluster.{row.formula_id.lower()}.v1", row)
        for row in executable
    )
    validate_adapter_registry(registrations)
    registration_by_formula = {row.formula_binding.formula_id: row for row in registrations}
    module_sha = _file_sha256(Path(__file__))
    release = SyntheticSuiteRelease(
        suite_id=config["package_id"],
        version=config["version"],
        release_sha256=canonical_sha256(
            {
                "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
                "generator_raw_sha256": module_sha,
                "source_receipt_raw_sha256": _file_sha256(
                    _repo_path(
                        next(
                            row["path"]
                            for row in config["upstream_bindings"]
                            if row["id"] == "XCOP_SOURCE_PREFLIGHT_RECEIPT"
                        )
                    )
                ),
            }
        ),
        ontology_sha256=catalogue.content_sha256,
        generator_sha256=module_sha,
        observation_operator_sha256=_json_sha256(
            {
                "noise_families": config["noise_families"],
                "noise": config["noise"],
                "scoring": config["scoring"],
            }
        ),
        changed_feature_ids=(
            "prediction.vector.acceleration",
            "response.vector.synthetic-acceleration",
            "truth.scalar.injection-id",
        ),
        change_level="MAJOR",
        response_calibrated=False,
        prediction_semantics_changed=True,
    )
    ledger = SyntheticReplayLedger("gravity.synthetic.xcop-source-shaped-matrix.v1", ())
    arrays: dict[str, np.ndarray] = {}
    candidate_cache: dict[tuple[str, str], dict[str, Any]] = {}
    numerical_failures: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for item in source_items:
        object_id = item["object_id"]
        for feature_id, value in item["values"].items():
            arrays[_array_key("source", object_id, feature_id)] = value
        scaffold_response = np.zeros((17, 17, 17, 3), dtype=np.float64)
        scaffold_variance = np.full(scaffold_response.shape, (_A0 * 1.0e-3) ** 2, dtype=np.float64)
        scaffold_id = f"cluster.spherical-lifted3d.{object_id.lower()}.execution-scaffold.v1"
        scaffold = _scenario(
            config=config,
            item=item,
            scenario_id=scaffold_id,
            truth_world_id="truth.execution-scaffold",
            nuisance_draw=0,
            response=scaffold_response,
            variance=scaffold_variance,
            truth_index=0,
        )
        validate_scenario_catalogue(scaffold, catalogue)
        for binding in executable:
            decision = decide_scenario_eligibility(binding, catalogue, scaffold)
            _require(
                decision.status is EligibilityStatus.ELIGIBLE,
                "executable cluster binding became ineligible",
            )
            try:
                result = execute_binding_in_process(
                    binding,
                    catalogue,
                    scaffold,
                    {key: item["values"][key] for key in binding.required_features},
                    {},
                )
                prediction = np.asarray(
                    result.output_values["prediction.vector.acceleration"], dtype=np.float64
                )
                _require(
                    prediction.shape == (17, 17, 17, 3) and np.all(np.isfinite(prediction)),
                    "candidate output invalid",
                )
                key = _array_key("candidate", object_id, binding.formula_id)
                arrays[key] = prediction
                candidate_cache[(object_id, binding.formula_id)] = {
                    "success": True,
                    "prediction": prediction,
                    "value_key": key,
                    "value_sha256": array_sha256(prediction),
                    "output_sha256": result.output_sha256,
                    "scaffold_scenario_sha256": scaffold.content_sha256,
                }
            except Exception as error:  # noqa: BLE001 - every adapter failure is retained
                failure = {
                    "object_id": object_id,
                    "formula_id": binding.formula_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                failure["failure_sha256"] = canonical_sha256(failure)
                numerical_failures.append(failure)
                candidate_cache[(object_id, binding.formula_id)] = {
                    "success": False,
                    "failure": failure,
                    "output_sha256": failure["failure_sha256"],
                    "scaffold_scenario_sha256": scaffold.content_sha256,
                }
        source_rows.append({"object_id": object_id, **item["source_metadata"]})

    scenario_rows: list[dict[str, Any]] = []
    confusion_counts: dict[str, dict[str, int]] = {
        truth: {candidate.formula_id: 0 for candidate in executable}
        for truth in config["mechanisms"]
    }
    truth_recovered_count = 0
    distinct_truth_recovered_count = 0
    candidate_comparison_count = 0
    dpel_block_count = 0
    for item in source_items:
        object_id = item["object_id"]
        for truth_index, truth_formula in enumerate(config["mechanisms"]):
            truth_cell = candidate_cache[(object_id, truth_formula)]
            if not truth_cell["success"]:
                numerical_failures.append(
                    {
                        "object_id": object_id,
                        "formula_id": truth_formula,
                        "stage": "truth-world-omitted-because-frozen-adapter-failed",
                        "failure_sha256": truth_cell["output_sha256"],
                    }
                )
                continue
            truth_prediction = truth_cell["prediction"]
            for nuisance_draw, family in enumerate(config["noise_families"]):
                truth_world_id = f"truth.{truth_formula.lower()}"
                scenario_id = (
                    f"cluster.spherical-lifted3d.{object_id.lower()}.{truth_world_id}."
                    f"noise-{family}.v1"
                )
                lineage = SeedLineage(
                    config["suite_seed"],
                    scenario_id,
                    object_id.lower(),
                    truth_world_id,
                    nuisance_draw,
                    0,
                )
                response, variance, noise_diagnostics = _noise_response(
                    truth_prediction,
                    family,
                    lineage,
                    item["source_metadata"]["combined_fractional_sigma"],
                    item["grid_radius_normalized"],
                    config,
                )
                scenario = _scenario(
                    config=config,
                    item=item,
                    scenario_id=scenario_id,
                    truth_world_id=truth_world_id,
                    nuisance_draw=nuisance_draw,
                    response=response,
                    variance=variance,
                    truth_index=truth_index,
                )
                truth_value = np.asarray([truth_index], dtype=np.int64)
                validate_scenario_catalogue(scenario, catalogue)
                validate_scenario_values(
                    scenario,
                    formula_values=item["values"],
                    response_values={"response.vector.synthetic-acceleration": response},
                    truth_values={"truth.scalar.injection-id": truth_value},
                    uncertainty_values={"synthetic-acceleration.diagonal-covariance": variance},
                )
                response_key = _array_key("response", object_id, truth_formula, family)
                variance_key = _array_key("variance", object_id, truth_formula, family)
                truth_key = _array_key("truth", object_id, truth_formula, family)
                arrays[response_key] = response
                arrays[variance_key] = variance
                arrays[truth_key] = truth_value
                comparisons: list[dict[str, Any]] = []
                for binding in executable:
                    cached = candidate_cache[(object_id, binding.formula_id)]
                    candidate_comparison_count += 1
                    if cached["success"]:
                        metric = _metrics(cached["prediction"], response, variance)
                        comparisons.append(
                            {
                                "candidate_formula_id": binding.formula_id,
                                "binding_sha256": binding.content_sha256,
                                "adapter_sha256": registration_by_formula[
                                    binding.formula_id
                                ].adapter_sha256,
                                "numerical_valid": True,
                                "metrics": metric,
                                "value_key": cached["value_key"],
                                "value_sha256": cached["value_sha256"],
                                "output_sha256": cached["output_sha256"],
                                "source_cache_scenario_sha256": cached["scaffold_scenario_sha256"],
                            }
                        )
                    else:
                        comparisons.append(
                            {
                                "candidate_formula_id": binding.formula_id,
                                "binding_sha256": binding.content_sha256,
                                "adapter_sha256": registration_by_formula[
                                    binding.formula_id
                                ].adapter_sha256,
                                "numerical_valid": False,
                                "metrics": None,
                                "failure": cached["failure"],
                                "output_sha256": cached["output_sha256"],
                                "source_cache_scenario_sha256": cached["scaffold_scenario_sha256"],
                            }
                        )
                valid_rows = [row for row in comparisons if row["numerical_valid"]]
                _require(valid_rows, "scenario has no numerically valid candidates")
                ordered = sorted(
                    valid_rows,
                    key=lambda row: (row["metrics"]["whitened_rmse"], row["candidate_formula_id"]),
                )
                minimum = ordered[0]["metrics"]["whitened_rmse"]
                tolerance = config["scoring"]["winner_absolute_tolerance"]
                winners = sorted(
                    row["candidate_formula_id"]
                    for row in ordered
                    if math.isclose(
                        row["metrics"]["whitened_rmse"], minimum, abs_tol=tolerance, rel_tol=0.0
                    )
                )
                second = ordered[1]["metrics"]["whitened_rmse"] if len(ordered) > 1 else None
                gap = 0.0 if second is None else float(second - minimum)
                distinct = (
                    len(winners) == 1
                    and gap >= config["scoring"]["minimum_whitened_gap_for_distinct_signature"]
                )
                truth_recovered = truth_formula in winners
                truth_recovered_count += int(truth_recovered)
                distinct_truth_recovered_count += int(truth_recovered and distinct)
                for winner in winners:
                    confusion_counts[truth_formula][winner] += 1
                completed_rows = []
                for binding in bindings:
                    decision = decide_scenario_eligibility(binding, catalogue, scenario)
                    ledger = ledger.append(
                        release=release,
                        binding=binding,
                        eligibility=decision,
                        adapter_sha256=(
                            registration_by_formula[binding.formula_id].adapter_sha256
                            if binding.formula_id in registration_by_formula
                            else None
                        ),
                        domain="cluster",
                        experiment_id=config["experiment_id"],
                    )
                    if decision.status is not EligibilityStatus.ELIGIBLE:
                        _require(
                            binding.formula_id == _DPEL
                            and decision.status is EligibilityStatus.UNADAPTED,
                            "unexpected adapter block",
                        )
                        dpel_block_count += 1
                        continue
                    comparison = next(
                        row
                        for row in comparisons
                        if row["candidate_formula_id"] == binding.formula_id
                    )
                    diagnostics = {
                        "candidate_formula_id": binding.formula_id,
                        "truth_formula_id": truth_formula,
                        "noise_family": family,
                        "real_response_used": False,
                        "source_cached_common_abi_execution": True,
                        "source_cache_scenario_sha256": comparison["source_cache_scenario_sha256"],
                        "winner_formula_ids": winners,
                        "whitened_gap": gap,
                        "distinct_by_frozen_threshold": distinct,
                    }
                    if comparison["numerical_valid"]:
                        status = status_from_result(
                            distinct_from_comparators=distinct,
                            self_injection_recovered=(
                                binding.formula_id == truth_formula and truth_recovered
                            ),
                            numerical_valid=True,
                            powered=gap
                            >= config["scoring"]["minimum_whitened_gap_for_distinct_signature"],
                        )
                        metric_payload = comparison["metrics"]
                        reasons = (
                            "response-blind",
                            "source-cached-common-abi-execution",
                            "synthetic-only",
                        )
                    else:
                        status = DiscoveryStatus.NUMERICAL_INVALID
                        metric_payload = {"numerical_valid": False}
                        diagnostics["failure"] = comparison["failure"]
                        reasons = ("numerical-invalid-retained", "response-blind", "synthetic-only")
                    ledger = ledger.complete_last_eligible(
                        release=release,
                        binding=binding,
                        adapter_sha256=registration_by_formula[binding.formula_id].adapter_sha256,
                        domain="cluster",
                        experiment_id=config["experiment_id"],
                        status=status,
                        scenario_id=scenario.scenario_id,
                        object_id=scenario.object_id,
                        truth_world_id=truth_world_id,
                        seed_lineage_sha256=canonical_sha256(scenario.seed_lineage.to_dict()),
                        nuisance_draw=nuisance_draw,
                        parameter_cell_id="frozen-no-free-parameters",
                        observable_ids=("response.vector.synthetic-acceleration",),
                        result_sha256=comparison["output_sha256"],
                        metrics_sha256=_json_sha256(metric_payload),
                        diagnostics_sha256=_json_sha256(diagnostics),
                        reason_codes=reasons,
                    )
                    completed_rows.append(
                        {
                            **comparison,
                            "discovery_status": status.value,
                            "completed_ledger_sequence": ledger.entries[-1].sequence,
                            "completed_ledger_entry_sha256": ledger.entries[-1].entry_sha256,
                        }
                    )
                scenario_rows.append(
                    {
                        "scenario": scenario.to_dict(),
                        "scenario_sha256": scenario.content_sha256,
                        "object_id": object_id,
                        "truth_formula_id": truth_formula,
                        "truth_world_id": truth_world_id,
                        "noise": noise_diagnostics,
                        "value_locators": {
                            "response": {
                                "path": VALUES_PATH.as_posix(),
                                "key": response_key,
                                "sha256": array_sha256(response),
                            },
                            "variance": {
                                "path": VALUES_PATH.as_posix(),
                                "key": variance_key,
                                "sha256": array_sha256(variance),
                            },
                            "truth": {
                                "path": VALUES_PATH.as_posix(),
                                "key": truth_key,
                                "sha256": array_sha256(truth_value),
                            },
                        },
                        "candidate_comparisons": completed_rows,
                        "injection_recovery": {
                            "primary_metric": "whitened_rmse",
                            "winner_formula_ids": winners,
                            "minimum_whitened_rmse": minimum,
                            "second_best_whitened_rmse": second,
                            "whitened_gap": gap,
                            "distinct_by_frozen_threshold": distinct,
                            "truth_recovered": truth_recovered,
                            "truth_distinctly_recovered": truth_recovered and distinct,
                        },
                    }
                )

    values_bytes = _npz_bytes(arrays)
    _require(values_bytes == _npz_bytes(arrays), "NPZ serialization is nondeterministic")
    scenarios_bytes = b"".join(_json_bytes(row) + b"\n" for row in scenario_rows)
    ledger_bytes = _json_bytes(ledger.to_dict(), indent=2)
    confusion = {
        "schema": "open-gravity-xcop-source-shaped-confusion-matrix-1.0",
        "truth_formula_ids": list(config["mechanisms"]),
        "candidate_formula_ids": [row.formula_id for row in executable],
        "winner_membership_counts": confusion_counts,
        "scenario_count": len(scenario_rows),
        "truth_recovered_count": truth_recovered_count,
        "distinct_truth_recovered_count": distinct_truth_recovered_count,
        "candidate_comparison_count": candidate_comparison_count,
        "numerical_failure_count": len(numerical_failures),
        "no_hand_ranking": True,
    }
    confusion_bytes = _json_bytes(confusion, indent=2)
    status = (
        "FROZEN_SYNTHETIC_ONLY_NUMERICAL_FAILURES_RETAINED_AWAITING_DISTINCT_AUDIT"
        if numerical_failures
        else "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT"
    )
    receipt_body = {
        "schema": "open-gravity-xcop-real-source-shaped-synthetic-injection-matrix-receipt-1.0",
        "package_id": config["package_id"],
        "version": config["version"],
        "status": status,
        "claim_class": config["claim_class"],
        "scientific_claim": "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION",
        "independent_audit_completed": False,
        "distinct_independent_audit_required": True,
        "cluster_count": len(source_items),
        "mechanism_count": len(executable),
        "noise_family_count": len(config["noise_families"]),
        "scenario_count": len(scenario_rows),
        "common_abi_execution_count": len(candidate_cache),
        "successful_common_abi_execution_count": sum(
            row["success"] for row in candidate_cache.values()
        ),
        "candidate_comparison_count": candidate_comparison_count,
        "confusion_matrix_cell_count": len(config["mechanisms"]) * len(executable),
        "replay_entry_count": len(ledger.entries),
        "dpel_adapter_block_count": dpel_block_count,
        "truth_recovered_count": truth_recovered_count,
        "distinct_truth_recovered_count": distinct_truth_recovered_count,
        "mechanism_ids": [row.formula_id for row in executable],
        "noise_family_ids": config["noise_families"],
        "source_clusters": source_rows,
        "source_files": opened_sources,
        "source_receipt_content_sha256": source_receipt["content_sha256"],
        "source_failures_retained": {
            "transport_source_blocked_clusters": source_receipt["adjudication"][
                "transport_source_blocked"
            ],
            "telegraph_source_blocked_clusters": source_receipt["adjudication"][
                "telegraph_source_blocked"
            ],
            "action_quarantined_clusters": source_receipt["adjudication"]["action_quarantined"],
            "adapter_blocks": config["adapter_blocks"],
        },
        "numerical_failures": numerical_failures,
        "formula_bindings": {row.formula_id: row.to_dict() for row in bindings},
        "formula_binding_sha256": {row.formula_id: row.content_sha256 for row in bindings},
        "adapter_sha256": {
            row.formula_binding.formula_id: row.adapter_sha256 for row in registrations
        },
        "release": release.to_dict(),
        "catalogue_sha256": catalogue.content_sha256,
        "package_hashes": {
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
            "module_raw_sha256": module_sha,
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH),
            "upstream_raw_sha256": _upstream_hashes(config),
        },
        "artifact_sha256": {
            "values.npz": hashlib.sha256(values_bytes).hexdigest(),
            "scenarios.jsonl": hashlib.sha256(scenarios_bytes).hexdigest(),
            "ledger.json": hashlib.sha256(ledger_bytes).hexdigest(),
            "confusion-matrix.json": hashlib.sha256(confusion_bytes).hexdigest(),
        },
        "access_accounting": {
            "source_density_files_opened": 8,
            "source_stellar_files_opened": 5,
            "unique_source_files_opened": 13,
            "source_bytes_opened": 308160,
            "pressure_files_opened": 0,
            "temperature_files_opened": 0,
            "hydrostatic_mass_columns_opened": 0,
            "lensing_files_opened": 0,
            "motion_files_opened": 0,
            "measured_response_rows_opened": 0,
            "scientific_scores_computed": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "limitations": [
            "The 17^3 field is a mass-normalized spherical lift of radial source profiles, not measured three-dimensional cluster structure.",
            "Synthetic recovery and confusion do not support or reject any gravity theory.",
            "AQUAL and QUMOND may be prediction-equivalent in spherical symmetry and must not be double-counted as independent evidence.",
            "GQNS is geometry-conditioned and may reduce to its spherical limit on this exact source lift.",
            "NFW parameters are derived only from the source field by the frozen adapter and are not fitted to a response.",
            "The correlated noise family stores a diagonal marginal covariance plus exact seed lineage, not a dense covariance matrix.",
            "DPEL remains adapter-blocked because a spherical cluster lift does not satisfy its disk-only geometry contract.",
        ],
    }
    receipt = {**receipt_body, "content_sha256": _json_sha256(receipt_body)}
    return receipt, values_bytes, scenarios_bytes, ledger_bytes, confusion_bytes


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing to overwrite changed artifact: {path}")
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
            _require(path.read_bytes() == payload, f"concurrent changed artifact exists: {path}")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def freeze() -> str:
    receipt, values, scenarios, ledger, confusion = derive_release()
    statuses = (
        _write_once(_ROOT / VALUES_PATH, values),
        _write_once(_ROOT / SCENARIOS_PATH, scenarios),
        _write_once(_ROOT / LEDGER_PATH, ledger),
        _write_once(_ROOT / CONFUSION_PATH, confusion),
        _write_once(_ROOT / RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    return ":".join(statuses)


def check() -> None:
    receipt, values, scenarios, ledger, confusion = derive_release()
    expected = (
        (_ROOT / VALUES_PATH, values),
        (_ROOT / SCENARIOS_PATH, scenarios),
        (_ROOT / LEDGER_PATH, ledger),
        (_ROOT / CONFUSION_PATH, confusion),
        (_ROOT / RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    for path, payload in expected:
        if not path.is_file() or path.read_bytes() != payload:
            raise SystemExit(f"stored X-COP synthetic artifact differs: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "check"))
    arguments = parser.parse_args()
    if arguments.command == "freeze":
        print(freeze())
    else:
        check()
        print("OK")
