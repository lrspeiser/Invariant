"""Response-blind full-3D PHANGS source and common-ABI adapter preflight.

The package reconstructs only the three frozen Lane-6 primary 17^3 source
cells.  Physics is delegated to the already frozen Lane-6 and DPEL01 solver
functions; this module supplies typed packet adaptation, provenance, and
target-free synthetic confusion accounting.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base
from sigma_theory_compiler import open_gravity_disk_polar_escape_load_v1 as dpel
from sigma_theory_compiler import open_gravity_lane6_same_grid_nonspherical_predictions_v1 as lane6
from sigma_theory_compiler import open_gravity_phangs_things_full3d_solver_bridge_v1 as bridge
from sigma_theory_compiler import (
    open_gravity_phangs_things_full3d_source_systematics_v1 as source_screen,
)
from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as source_builder,
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

CONFIG_PATH = Path("configs/open_gravity_full3d_phangs_synthetic_adapter_preflight_v1.json")
OUTPUT_DIR = Path("runs/gravity/open-gravity-full3d-phangs-synthetic-adapter-preflight-v1")
PACKETS_PATH = OUTPUT_DIR / "packets.jsonl"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT = "galaxy.full3d-synthetic-preflight.v1"
_A0 = 1.2e-10
_MSUN_PC3_TO_G_CM3 = 6.768109983980883e-23

_MECHANISM_ENTRYPOINTS = {
    "AQUAL_SIMPLE_MU": "lane6_aqual_adapter",
    "DPEL01_DISK_POLAR_ESCAPE_LOAD": "dpel01_full3d_adapter",
    "GP01_ELLIPTIC_N2_L035": "lane6_gp01_adapter",
    "GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE": "lane6_gqns_adapter",
    "MASHHOON_RAHVAR_NLG_Q0": "lane6_published_nonlocal_adapter",
    "NEWTON": "lane6_newton_adapter",
    "NFW_SOURCE_MATCHED_CONTROL": "lane6_nfw_adapter",
    "QUMOND_SIMPLE_NU": "lane6_qumond_adapter",
    "REFRACTED_GRAVITY_DISKMASS_MEDIAN": "lane6_refracted_adapter",
}

_FORMULA_FEATURES = (
    "geometry.scalar.grid-spacing-normalized",
    "geometry.scalar.half-box-length",
    "geometry.scalar.x-coordinate",
    "geometry.scalar.y-coordinate",
    "geometry.scalar.z-coordinate",
    "geometry.vector.disk-normal",
    "source.scalar.expected-mass-normalized",
    "source.scalar.mass-density",
    "source.scalar.solver-density",
)


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
        raise SchemaViolation("full3d preflight path escaped repository")
    path = (_ROOT / parsed.as_posix()).resolve()
    if not path.is_relative_to(_ROOT):
        raise SchemaViolation("full3d preflight path escaped repository")
    return path


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def _raw_fits_inventory(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    contract = config["raw_fits_inventory"]
    acquisition = json.loads(_repo_path(contract["acquisition_receipt_path"]).read_text())
    rows = tuple(
        {
            "id": row["id"],
            "path": row["relative_private_path"],
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in acquisition["inventory"]
    )
    root = hashlib.sha256(_json_bytes(list(rows))).hexdigest()
    if (
        len(rows) != contract["count"]
        or sum(row["bytes"] for row in rows) != contract["bytes"]
        or root != contract["root_sha256"]
    ):
        raise SchemaViolation("raw FITS inventory contract changed")
    for row in rows:
        path = _repo_path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise SchemaViolation(f"raw FITS source missing or resized: {row['id']}")
        if _file_sha256(path) != row["sha256"]:
            raise SchemaViolation(f"raw FITS source changed: {row['id']}")
    return rows


def validate_config(config: Mapping[str, Any], *, verify_sources: bool = True) -> None:
    expected = {
        "schema",
        "package_id",
        "version",
        "status",
        "claim_class",
        "experiment_id",
        "objects",
        "primary_cell_id",
        "grid_nodes",
        "a0_m_s2",
        "pc_m",
        "source_bindings",
        "raw_fits_inventory",
        "mechanisms",
        "dpel01_frozen_adapter",
        "parameter_schema_path",
        "output_directory",
        "access_contract",
    }
    if set(config) != expected:
        raise SchemaViolation("full3d preflight config keys changed")
    if config["schema"] != "open-gravity-full3d-phangs-synthetic-adapter-preflight-1.0":
        raise SchemaViolation("full3d preflight schema changed")
    if config["claim_class"] != "SYNTHETIC_DIRECTIONAL_SIGNAL":
        raise SchemaViolation("full3d preflight claim ceiling changed")
    if config["experiment_id"] != _EXPERIMENT:
        raise SchemaViolation("full3d preflight experiment changed")
    if tuple(config["objects"]) != ("NGC2903", "NGC3351", "NGC3627"):
        raise SchemaViolation("full3d preflight object inventory changed")
    if config["grid_nodes"] != 17 or tuple(config["mechanisms"]) != tuple(
        sorted(_MECHANISM_ENTRYPOINTS)
    ):
        raise SchemaViolation("full3d preflight grid or mechanism inventory changed")
    if float(config["a0_m_s2"]) != _A0 or float(config["pc_m"]) <= 0.0:
        raise SchemaViolation("full3d preflight physical normalization changed")
    expected_dpel = {
        "D_parallel_over_D0": "0.5",
        "D_perpendicular_over_D0": "1.5",
        "chi_over_D0": "0.08",
        "source_gain": "1.0",
        "Gamma0_L0_squared_over_D0": "0.2",
        "beta_L0_squared_over_D0": "0.35",
        "dt": "0.001",
        "steps": 100,
        "boundary": (
            "periodic target-free preflight inherited from DPEL01; "
            "isolated empirical boundary remains blocked"
        ),
    }
    if config["dpel01_frozen_adapter"] != expected_dpel:
        raise SchemaViolation("DPEL01 frozen adapter parameters changed")
    if _repo_path(config["output_directory"]) != (_ROOT / OUTPUT_DIR).resolve():
        raise SchemaViolation("full3d preflight output directory changed")
    access = config["access_contract"]
    if any(access.values()):
        raise SchemaViolation("full3d preflight response-blind boundary changed")
    if verify_sources:
        for binding in config["source_bindings"]:
            if set(binding) != {"path", "sha256"}:
                raise SchemaViolation("source binding schema changed")
            path = _repo_path(binding["path"])
            if not path.is_file() or _file_sha256(path) != binding["sha256"]:
                raise SchemaViolation(f"source binding changed: {binding['path']}")
        _raw_fits_inventory(config)


def _catalogue(config: Mapping[str, Any]):
    provenance = canonical_sha256(config["source_bindings"])
    specs = (
        ("geometry.scalar.grid-spacing-normalized", "normalized grid spacing", 0, "1", ("object",)),
        ("geometry.scalar.half-box-length", "solver half-box length", 0, "m", ("object",)),
        ("geometry.scalar.x-coordinate", "x coordinate", 0, "m", ("x",)),
        ("geometry.scalar.y-coordinate", "y coordinate", 0, "m", ("y",)),
        ("geometry.scalar.z-coordinate", "z coordinate", 0, "m", ("z",)),
        ("geometry.vector.disk-normal", "disk normal", 1, "1", ("component",)),
        (
            "source.scalar.expected-mass-normalized",
            "normalized enclosed source mass",
            0,
            "1",
            ("object",),
        ),
        ("source.scalar.mass-density", "mass density", 0, "kg m^-3", ("x", "y", "z")),
        (
            "source.scalar.solver-density",
            "Lane6 normalized source density",
            0,
            "1",
            ("x", "y", "z"),
        ),
        (
            "response.vector.synthetic-acceleration",
            "synthetic acceleration target",
            1,
            "m s^-2",
            ("x", "y", "z", "component"),
        ),
        (
            "prediction.vector.acceleration",
            "predicted acceleration",
            1,
            "m s^-2",
            ("x", "y", "z", "component"),
        ),
        (
            "truth.scalar.injection-id",
            "synthetic injection identity",
            0,
            "typed hidden value",
            ("object",),
        ),
    )
    elements = []
    for element_id, quantity, rank, unit, axes in specs:
        if element_id.startswith("response."):
            role = DataRole.SCORING_ONLY_RESPONSE
        elif element_id.startswith("prediction."):
            role = DataRole.DERIVED
        elif element_id.startswith("truth."):
            role = DataRole.LATENT_SYNTHETIC_TRUTH
        else:
            role = DataRole.FORMULA_INPUT
        dimensions = {
            "1": (0, 0, 0, 0, 0, 0, 0),
            "typed hidden value": (0, 0, 0, 0, 0, 0, 0),
            "m": (0, 1, 0, 0, 0, 0, 0),
            "kg m^-3": (1, -3, 0, 0, 0, 0, 0),
            "m s^-2": (0, 1, -2, 0, 0, 0, 0),
        }[unit]
        elements.append(
            DataElement(
                element_id=element_id,
                namespace=element_id.rsplit(".", 1)[0],
                physical_quantity=quantity,
                tensor_rank=rank,
                si_dimension=dimensions,
                canonical_unit=unit,
                frame="solver-source" if not element_id.startswith("truth.") else "latent",
                support="17^3 primary Lane6 grid",
                axes=axes,
                component="total",
                derivation_parents=(),
                uncertainty=(
                    UncertaintyKind.COVARIANCE
                    if element_id.startswith(("response.", "prediction."))
                    else UncertaintyKind.NONE
                ),
                availability=(
                    Availability.SYNTHETIC_ONLY
                    if element_id.startswith(("response.", "prediction.", "truth."))
                    else Availability.PUBLIC_SOURCE
                ),
                experiment_roles=(ExperimentRole(_EXPERIMENT, role),),
                provenance_sha256=provenance,
            )
        )
    return catalogue_from_elements("open-gravity-full3d-phangs-preflight", "v1.0.0", elements)


def _primary_source_items(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    lane_config = lane6.load_config()
    source_receipt = lane6.validate_bindings(lane_config)
    lane_receipt = json.loads(
        _repo_path(
            "runs/gravity/open-gravity-lane6-same-grid-nonspherical-predictions-v1/receipt.json"
        ).read_text()
    )
    primary_predictions = {row["object_id"]: row for row in lane_receipt["primary_predictions"]}
    source_config, acquisition, source_private = source_screen._load_source_builder_evidence()
    bridge_config = bridge.load_config()
    source_paths = source_builder._source_paths(acquisition)
    metadata_by_id = {row["object_id"]: row for row in source_config["objects"]}
    private_by_id = {row["object_id"]: row for row in source_private["objects"]}
    sealed = {(row["object_id"], row["cell_id"]): row for row in source_receipt["cells"]}
    rows = []
    for object_id in config["objects"]:
        metadata = metadata_by_id[object_id]
        images = source_builder._load_object_images(object_id, source_paths)
        maps = source_builder._surface_maps(
            source_config,
            metadata,
            images,
            n=256,
            box_kpc=40.0,
            beam="ROBUST_PRIMARY",
            use_sip=False,
        )
        rhalf_pc = source_builder._half_mass_radius_pc(
            maps["stellar_fixed"], maps["x_pc"], maps["y_pc"], float(maps["dx_pc"])
        )
        cell_id = str(config["primary_cell_id"])
        if private_by_id[object_id]["primary_cell_id"] != cell_id:
            raise SchemaViolation("source-builder primary cell changed")
        result = lane6._density_variant(
            bridge_config,
            maps,
            stellar_surface=maps["stellar_fixed"],
            co_surface=maps["co"],
            hstar_pc=(rhalf_pc / 1.678) * 0.136986301369863,
            hgas_pc=200.0,
            half_box_kpc=30.0,
            nodes=17,
        )
        density_hash = bridge.array_sha256(result["density"])
        source_cell = sealed[(object_id, cell_id)]
        if (
            density_hash != source_cell["field_hashes"]["density"]
            or density_hash != primary_predictions[object_id]["density_hash"]
        ):
            raise SchemaViolation("reconstructed primary Lane6 source grid changed")
        rows.append(
            {
                "object_id": object_id,
                "cell_id": cell_id,
                "sealed_density_sha256": density_hash,
                "predecessor_source_gates_pass": source_cell["all_numerical_gates_pass"],
                "sealed": source_cell,
                **result,
            }
        )
    return tuple(rows)


def _adapter_inputs(features: Mapping[str, Any]) -> dict[str, Any]:
    if set(features) != set(_FORMULA_FEATURES):
        raise SchemaViolation("full3d adapter feature projection changed")
    density = np.asarray(features["source.scalar.solver-density"], dtype=np.float64)
    physical_kg_m3 = np.asarray(features["source.scalar.mass-density"], dtype=np.float64)
    if density.shape != (17, 17, 17) or physical_kg_m3.shape != density.shape:
        raise SchemaViolation("full3d adapter source shape changed")
    grid = base.make_grid(17)
    spacing = float(np.asarray(features["geometry.scalar.grid-spacing-normalized"])[0])
    if not math.isclose(spacing, grid.spacing, abs_tol=0.0, rel_tol=0.0):
        raise SchemaViolation("full3d adapter grid spacing changed")
    return {
        "density": density,
        "physical_density_g_cm3": physical_kg_m3 / 1000.0,
        "grid": grid,
        "mass": float(np.asarray(features["source.scalar.expected-mass-normalized"])[0]),
        "half_box_kpc": float(np.asarray(features["geometry.scalar.half-box-length"])[0])
        / (1000.0 * 3.085677581491367e16),
        "disk_normal": np.asarray(features["geometry.vector.disk-normal"], dtype=np.float64),
    }


def _newton_state(values: Mapping[str, Any]):
    grid = values["grid"]
    boundary = lane6._newton_boundary(grid, values["mass"])
    result = base.solve_poisson(4.0 * math.pi * values["density"], boundary, grid.spacing)
    return boundary, result


def _vector_field(components: Sequence[np.ndarray]) -> np.ndarray:
    value = np.stack(tuple(components), axis=-1).astype(np.float64) * _A0
    if value.shape != (17, 17, 17, 3) or not np.all(np.isfinite(value)):
        raise SchemaViolation("full3d adapter emitted an invalid acceleration field")
    return value


def _lane6_field(mechanism: str, features: Mapping[str, Any]) -> Mapping[str, Any]:
    values = _adapter_inputs(features)
    density = values["density"]
    grid = values["grid"]
    boundary, newton = _newton_state(values)
    if mechanism == "NEWTON":
        components = base.acceleration(newton.potential, grid.spacing)
    elif mechanism == "NFW_SOURCE_MATCHED_CONTROL":
        rms, _, _ = lane6._rms_geometry(density, grid)
        potential, _ = lane6._solve_nfw_control(
            density, grid, baryonic_mass=values["mass"], rms_radius=rms
        )
        components = base.acceleration(potential, grid.spacing)
    elif mechanism == "AQUAL_SIMPLE_MU":
        mond_boundary = bridge.spherical_boundary(
            grid, values["mass"], mond=True, integration_samples=100_000
        )
        result = base.solve_aqual(
            4.0 * math.pi * density,
            mond_boundary,
            grid.spacing,
            a0=1.0,
            mu_floor=1.0e-6,
            damping=0.5,
            max_iterations=500,
            delta_tolerance=1.0e-8,
            residual_tolerance=2.0e-7,
        )
        if not result.converged:
            raise SchemaViolation("frozen Lane6 AQUAL solver did not converge")
        components = base.acceleration(result.potential, grid.spacing)
    elif mechanism == "QUMOND_SIMPLE_NU":
        mond_boundary = bridge.spherical_boundary(
            grid, values["mass"], mond=True, integration_samples=100_000
        )
        _, result, _ = base.solve_qumond(
            4.0 * math.pi * density,
            boundary,
            mond_boundary,
            grid.spacing,
            a0=1.0,
            nu_floor=1.0e-6,
        )
        components = base.acceleration(result.potential, grid.spacing)
    elif mechanism == "REFRACTED_GRAVITY_DISKMASS_MEDIAN":
        potential, _ = lane6._solve_refracted(
            density, values["physical_density_g_cm3"], boundary, grid
        )
        components = base.acceleration(potential, grid.spacing)
    elif mechanism == "GP01_ELLIPTIC_N2_L035":
        potential, _ = lane6._solve_gp01(density, newton.potential, boundary, grid)
        components = base.acceleration(potential, grid.spacing)
    elif mechanism == "GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE":
        potential, _, _ = lane6._solve_gqns(density, grid)
        components = base.acceleration(potential, grid.spacing)
    elif mechanism == "MASHHOON_RAHVAR_NLG_Q0":
        targets = np.stack((grid.x.ravel(), grid.y.ravel(), grid.z.ravel()), axis=1)
        correction = lane6._nlg_correction(
            density, grid, targets, half_box_kpc=values["half_box_kpc"]
        ).reshape(17, 17, 17, 3)
        components = tuple(
            component + correction[..., index]
            for index, component in enumerate(base.acceleration(newton.potential, grid.spacing))
        )
    else:
        raise SchemaViolation("unknown frozen Lane6 mechanism")
    return {"prediction.vector.acceleration": _vector_field(components)}


def _no_parameters(parameters: Mapping[str, Any]) -> None:
    if parameters:
        raise SchemaViolation("frozen full3d adapters take no free parameters")


def lane6_newton_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    return _lane6_field("NEWTON", features)


def lane6_nfw_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    return _lane6_field("NFW_SOURCE_MATCHED_CONTROL", features)


def lane6_aqual_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    return _lane6_field("AQUAL_SIMPLE_MU", features)


def lane6_qumond_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    return _lane6_field("QUMOND_SIMPLE_NU", features)


def lane6_refracted_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    return _lane6_field("REFRACTED_GRAVITY_DISKMASS_MEDIAN", features)


def lane6_gp01_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    return _lane6_field("GP01_ELLIPTIC_N2_L035", features)


def lane6_gqns_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    return _lane6_field("GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE", features)


def lane6_published_nonlocal_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    return _lane6_field("MASHHOON_RAHVAR_NLG_Q0", features)


def dpel01_full3d_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]):
    _no_parameters(parameters)
    values = _adapter_inputs(features)
    density = values["density"]
    grid = values["grid"]
    source = density / float(np.max(density))
    shape = dpel.disk_shape_from_density(source, grid.spacing)
    supplied_normal = values["disk_normal"] / np.linalg.norm(values["disk_normal"])
    if abs(float(np.dot(shape["normal"], supplied_normal))) < 1.0 - 1.0e-12:
        raise SchemaViolation("DPEL01 disk-normal packet differs from frozen source functional")
    tensor = dpel.diffusion_tensor(shape["normal"], shape["activation"], 0.5, 1.5)
    load, _ = dpel.integrate_load(
        source,
        grid.spacing,
        tensor,
        chi=0.08,
        source_gain=1.0,
        gamma0=0.2,
        beta=0.35,
        dt=0.001,
        steps=100,
    )
    _, newton = _newton_state(values)
    baryonic = np.stack(base.acceleration(newton.potential, grid.spacing), axis=-1)
    load_acceleration = np.moveaxis(dpel.matter_acceleration_from_load(load, grid.spacing), 0, -1)
    return {"prediction.vector.acceleration": (baryonic + load_acceleration) * _A0}


def _bindings(config: Mapping[str, Any]) -> tuple[FormulaExecutionBinding, ...]:
    schema_path = _repo_path(config["parameter_schema_path"])
    schema_sha = _file_sha256(schema_path)
    lane_sha = _file_sha256(
        _repo_path(
            "src/sigma_theory_compiler/open_gravity_lane6_same_grid_nonspherical_predictions_v1.py"
        )
    )
    dpel_sha = _file_sha256(
        _repo_path("src/sigma_theory_compiler/open_gravity_disk_polar_escape_load_v1.py")
    )
    result = []
    for mechanism in config["mechanisms"]:
        result.append(
            FormulaExecutionBinding(
                binding_id=f"binding.full3d.{mechanism.lower()}.v1",
                formula_id=mechanism,
                formula_version="v1.0.0-frozen-adapter",
                formula_sha256=(
                    dpel_sha if mechanism == "DPEL01_DISK_POLAR_ESCAPE_LOAD" else lane_sha
                ),
                status=BindingStatus.EXECUTABLE,
                entrypoint=(
                    "sigma_theory_compiler."
                    "open_gravity_full3d_phangs_synthetic_adapter_preflight_v1:"
                    f"{_MECHANISM_ENTRYPOINTS[mechanism]}"
                ),
                required_features=_FORMULA_FEATURES,
                optional_features=(),
                emitted_features=("prediction.vector.acceleration",),
                domains=("galaxy",),
                geometry_support=("nonspherical3d",),
                time_support=("static",),
                parameter_schema_path=config["parameter_schema_path"],
                parameter_schema_sha256=schema_sha,
                approximation_ceiling="response-blind 17^3 source and adapter preflight only",
                health_gates=("determinism", "finite-output", "source-hash", "typed-output"),
                resource_bounds=ResourceBounds(180, 2_000_000_000, 2_000_000),
            )
        )
    return tuple(result)


def _feature_values(item: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    grid = item["grid"]
    half_box_m = item["half_box_kpc"] * 1000.0 * float(config["pc_m"])
    coordinates = grid.coordinates * half_box_m
    normal = dpel.disk_shape_from_density(item["density"], grid.spacing)["normal"].astype(
        np.float64
    )
    return {
        "geometry.scalar.grid-spacing-normalized": np.asarray([grid.spacing], dtype=np.float64),
        "geometry.scalar.half-box-length": np.asarray([half_box_m], dtype=np.float64),
        "geometry.scalar.x-coordinate": coordinates.astype(np.float64),
        "geometry.scalar.y-coordinate": coordinates.astype(np.float64),
        "geometry.scalar.z-coordinate": coordinates.astype(np.float64),
        "geometry.vector.disk-normal": normal,
        "source.scalar.expected-mass-normalized": np.asarray(
            [item["expected_mass"]], dtype=np.float64
        ),
        "source.scalar.mass-density": (
            np.asarray(item["physical_density_g_cm3"], dtype=np.float64) * 1000.0
        ),
        "source.scalar.solver-density": np.asarray(item["density"], dtype=np.float64),
    }


def _scenario(
    item: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
    response: np.ndarray,
    variance: np.ndarray,
    truth_index: int,
    config: Mapping[str, Any],
) -> ScenarioDescriptor:
    object_key = item["object_id"].lower()
    scenario_id = f"galaxy.full3d.{object_key}.primary.v1"
    path = "packets.jsonl"
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
    return ScenarioDescriptor(
        scenario_id=scenario_id,
        object_id=object_key,
        experiment_id=_EXPERIMENT,
        domain="galaxy",
        geometry_mode="nonspherical3d",
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
                path,
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
                path,
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
                path,
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
                path,
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
                path,
                array_sha256(variance),
            ),
        ),
        anchors=(
            AnchorBinding(
                "lane6.primary-source-receipt.v1",
                "runs/gravity/open-gravity-phangs-things-full3d-source-systematics-v1/receipt.json",
                "7b84da7ff79cb4d482c57a0b99399c9e61610737fde8ad02fb976a69dd2a16cd",
            ),
            AnchorBinding(
                "phangs.things.source-receipt.v1",
                "runs/gravity/open-gravity-phangs-things-model-lifted-3d-source-builder-v1/receipt.json",
                "18c7d20b1ff7413edbd1b0b9277c53e6cbe4d9d05b4bcf55a8c70bc054f8ea82",
            ),
        ),
        seed_lineage=SeedLineage(
            170317,
            scenario_id,
            object_key,
            "newton-known-answer",
            0,
            0,
        ),
    )


def _confusion(predictions: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for truth_id in sorted(predictions):
        truth = predictions[truth_id]
        scale = max(float(np.sqrt(np.mean(truth * truth))), 1.0e-30)
        distances = []
        for candidate_id in sorted(predictions):
            relative_rmse = float(
                np.sqrt(np.mean((predictions[candidate_id] - truth) ** 2)) / scale
            )
            distances.append({"candidate_id": candidate_id, "relative_rmse": relative_rmse})
        minimum = min(row["relative_rmse"] for row in distances)
        winners = sorted(
            row["candidate_id"]
            for row in distances
            if math.isclose(row["relative_rmse"], minimum, abs_tol=1.0e-15, rel_tol=0.0)
        )
        rows.append(
            {
                "truth_id": truth_id,
                "candidate_distances": distances,
                "winner_ids": winners,
                "self_recovered": truth_id in winners,
            }
        )
    return rows


def derive_release() -> tuple[dict[str, Any], bytes, bytes]:
    config = load_config()
    validate_config(config)
    catalogue = _catalogue(config)
    bindings = _bindings(config)
    validate_binding_catalogue(bindings, catalogue)
    registrations = tuple(
        AdapterRegistration.create(f"adapter.full3d.{row.formula_id.lower()}.v1", row)
        for row in bindings
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
                "generator_sha256": module_sha,
                "raw_fits_inventory_root_sha256": config["raw_fits_inventory"]["root_sha256"],
            }
        ),
        ontology_sha256=catalogue.content_sha256,
        generator_sha256=module_sha,
        observation_operator_sha256=canonical_sha256(
            {"known_answer": "NEWTON", "response_noise": "ZERO"}
        ),
        changed_feature_ids=tuple(sorted((*_FORMULA_FEATURES, "prediction.vector.acceleration"))),
        change_level="MINOR",
        response_calibrated=False,
        prediction_semantics_changed=True,
    )
    ledger = SyntheticReplayLedger("gravity.synthetic.full3d-phangs-preflight.replays", ())
    packet_rows = []
    source_items = _primary_source_items(config)
    for truth_index, item in enumerate(source_items):
        feature_values = _feature_values(item, config)
        newton_response = np.asarray(
            lane6_newton_adapter(feature_values, {})["prediction.vector.acceleration"],
            dtype=np.float64,
        )
        variance = np.full(newton_response.shape, (_A0 * 1.0e-3) ** 2, dtype=np.float64)
        scenario = _scenario(item, feature_values, newton_response, variance, truth_index, config)
        validate_scenario_catalogue(scenario, catalogue)
        truth_value = np.asarray([truth_index], dtype=np.int64)
        validate_scenario_values(
            scenario,
            formula_values=feature_values,
            response_values={"response.vector.synthetic-acceleration": newton_response},
            truth_values={"truth.scalar.injection-id": truth_value},
            uncertainty_values={"synthetic-acceleration.diagonal-covariance": variance},
        )
        predictions = {}
        prediction_rows = []
        for binding in bindings:
            decision = decide_scenario_eligibility(binding, catalogue, scenario)
            if decision.status is not EligibilityStatus.ELIGIBLE:
                raise SchemaViolation("frozen full3d adapter became ineligible")
            result = execute_binding_in_process(
                binding,
                catalogue,
                scenario,
                {element_id: feature_values[element_id] for element_id in _FORMULA_FEATURES},
                {},
            )
            prediction = result.output_values["prediction.vector.acceleration"]
            predictions[binding.formula_id] = prediction
            registration = registration_by_formula[binding.formula_id]
            metrics = {
                "relative_rmse_to_newton_known_answer": float(
                    np.sqrt(np.mean((prediction - newton_response) ** 2))
                    / max(float(np.sqrt(np.mean(newton_response**2))), 1.0e-30)
                )
            }
            diagnostics = {
                "source_adapter_preflight": True,
                "real_response_used": False,
                "mechanism_id": binding.formula_id,
            }
            ledger = ledger.append(
                release=release,
                binding=binding,
                eligibility=decision,
                adapter_sha256=registration.adapter_sha256,
                domain="galaxy",
                experiment_id=_EXPERIMENT,
            )
            ledger = ledger.complete_last_eligible(
                release=release,
                binding=binding,
                adapter_sha256=registration.adapter_sha256,
                domain="galaxy",
                experiment_id=_EXPERIMENT,
                status=DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR,
                scenario_id=scenario.scenario_id,
                object_id=scenario.object_id,
                truth_world_id="newton-known-answer",
                seed_lineage_sha256=canonical_sha256(scenario.seed_lineage.to_dict()),
                nuisance_draw=0,
                parameter_cell_id="frozen-no-free-parameters",
                observable_ids=("response.vector.synthetic-acceleration",),
                result_sha256=result.output_sha256,
                metrics_sha256=_json_sha256(metrics),
                diagnostics_sha256=_json_sha256(diagnostics),
                reason_codes=("source-adapter-preflight", "synthetic-known-answer-only"),
            )
            prediction_rows.append(
                {
                    "mechanism_id": binding.formula_id,
                    "binding_sha256": binding.content_sha256,
                    "adapter_sha256": registration.adapter_sha256,
                    "artifact": result.output_predictions[
                        "prediction.vector.acceleration"
                    ].to_dict(),
                    "value": prediction.tolist(),
                    "output_sha256": result.output_sha256,
                    "metrics": metrics,
                }
            )
        confusion = _confusion(predictions)
        packet_rows.append(
            {
                "object_id": item["object_id"],
                "cell_id": item["cell_id"],
                "sealed_density_sha256": item["sealed_density_sha256"],
                "predecessor_source_gates_pass": item["predecessor_source_gates_pass"],
                "scenario": scenario.to_dict(),
                "values": {
                    **{key: value.tolist() for key, value in feature_values.items()},
                    "response.vector.synthetic-acceleration": newton_response.tolist(),
                    "truth.scalar.injection-id": truth_value.tolist(),
                    "uncertainty.synthetic-acceleration-variance": variance.tolist(),
                },
                "predictions": prediction_rows,
                "target_free_confusion": confusion,
            }
        )
    packets = b"".join(_json_bytes(row) + b"\n" for row in packet_rows)
    ledger_bytes = _json_bytes(ledger.to_dict(), indent=2)
    raw_inventory = _raw_fits_inventory(config)
    self_recoveries = [
        row["self_recovered"] for packet in packet_rows for row in packet["target_free_confusion"]
    ]
    receipt_body = {
        "schema": "open-gravity-full3d-phangs-synthetic-adapter-preflight-receipt-1.0",
        "package_id": config["package_id"],
        "version": config["version"],
        "status": "PASS_SOURCE_AND_NINE_COMMON_ABI_ADAPTER_PREFLIGHT_TARGET_FREE_CONFUSION_ONLY",
        "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "scientific_claim": "NONE_SOURCE_ADAPTER_AND_POWER_STEERING_ONLY",
        "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
        "module_raw_sha256": module_sha,
        "catalogue_sha256": catalogue.content_sha256,
        "release": release.to_dict(),
        "source_bindings": config["source_bindings"],
        "raw_fits_inventory": {
            **config["raw_fits_inventory"],
            "verified_rows": list(raw_inventory),
        },
        "object_count": len(packet_rows),
        "primary_source_cell_count": len(packet_rows),
        "grid_shape": [17, 17, 17],
        "mechanism_count": len(bindings),
        "mechanism_ids": [row.formula_id for row in bindings],
        "adapter_executions": len(packet_rows) * len(bindings),
        "replay_entry_count": len(ledger.entries),
        "target_free_confusion_cells": len(packet_rows) * len(bindings) ** 2,
        "target_free_self_injections_recovered": sum(self_recoveries),
        "target_free_self_injections_total": len(self_recoveries),
        "packets_jsonl_sha256": hashlib.sha256(packets).hexdigest(),
        "ledger_json_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "source_density_hashes": {
            row["object_id"]: row["sealed_density_sha256"] for row in packet_rows
        },
        "predecessor_source_gate_failures_retained": sum(
            not row["predecessor_source_gates_pass"] for row in packet_rows
        ),
        "access_accounting": {
            "scientific_response_files_opened": 0,
            "scientific_response_rows_opened": 0,
            "lensing_response_files_opened": 0,
            "real_scores_computed": 0,
            "response_calibrated": False,
            "source_fits_files_verified_and_opened": len(raw_inventory),
            "source_fits_bytes_verified_and_opened": sum(row["bytes"] for row in raw_inventory),
        },
        "blocks": [
            "no velocity or lensing response opened",
            "model-lifted 2.5D source in a full-3D field grid is not measured 3D mass",
            "DPEL01 retains its target-free periodic boundary; isolated empirical boundary remains blocked",
            "DGKT01 finite transport remains blocked on angular convergence",
            "synthetic confusion cannot support or reject a gravity theory",
            "the retained NGC2903 predecessor primary-source numerical gate failure remains visible",
        ],
    }
    receipt = {**receipt_body, "content_sha256": _json_sha256(receipt_body)}
    return receipt, packets, ledger_bytes


def _write_identical(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise SchemaViolation(f"existing full3d preflight artifact differs: {path.name}")
        return "EXISTING_IDENTICAL"
    path.write_bytes(payload)
    return "CREATED"


def build() -> str:
    receipt, packets, ledger = derive_release()
    states = (
        _write_identical(_ROOT / PACKETS_PATH, packets),
        _write_identical(_ROOT / LEDGER_PATH, ledger),
        _write_identical(_ROOT / RECEIPT_PATH, _json_bytes(receipt, indent=2)),
    )
    return ":".join(states)


def check() -> str:
    receipt, packets, ledger = derive_release()
    if (_ROOT / PACKETS_PATH).read_bytes() != packets:
        raise SchemaViolation("stored full3d packets differ from deterministic replay")
    if (_ROOT / LEDGER_PATH).read_bytes() != ledger:
        raise SchemaViolation("stored full3d ledger differs from deterministic replay")
    if json.loads((_ROOT / RECEIPT_PATH).read_text()) != receipt:
        raise SchemaViolation("stored full3d receipt differs from deterministic replay")
    return "VALID"


__all__ = [
    "CONFIG_PATH",
    "LEDGER_PATH",
    "PACKETS_PATH",
    "RECEIPT_PATH",
    "build",
    "check",
    "derive_release",
    "dpel01_full3d_adapter",
    "lane6_aqual_adapter",
    "lane6_gp01_adapter",
    "lane6_gqns_adapter",
    "lane6_newton_adapter",
    "lane6_nfw_adapter",
    "lane6_published_nonlocal_adapter",
    "lane6_qumond_adapter",
    "lane6_refracted_adapter",
    "load_config",
    "validate_config",
]
