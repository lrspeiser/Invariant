"""Response-blind lens/ray/clock source-shaped synthetic injection matrix."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
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
    EligibilityStatus,
    FormulaExecutionBinding,
    ResourceBounds,
    validate_binding_catalogue,
)
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
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
    "configs/open_gravity_lens_ray_clock_source_shaped_synthetic_injection_matrix_v2.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_lens_ray_clock_source_shaped_synthetic_injection_matrix_v2.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-lens-ray-clock-source-shaped-synthetic-injection-matrix-v2"
)
VALUES_PATH = OUTPUT_DIR / "values.npz"
SCENARIOS_PATH = OUTPUT_DIR / "scenarios.jsonl"
LEDGER_PATH = OUTPUT_DIR / "ledger.json"
CONFUSION_PATH = OUTPUT_DIR / "confusion-matrix.json"
DIAGNOSTICS_PATH = OUTPUT_DIR / "invariance-and-identifiability.json"
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]

_FEATURES = tuple(
    sorted(
        (
            "source.scalar.einstein-mass-kg",
            "source.scalar.geometric-time-delay-days",
            "source.scalar.gr-time-delay-days",
            "source.scalar.gravitational-radius-mpc",
            "source.scalar.image-time-delay-available",
            "source.scalar.image-time-delay-days",
            "source.scalar.image-time-delay-uncertainty-days",
            "source.scalar.lens-code",
            "source.scalar.path-age-mpc",
            "source.scalar.stellar-mass-fraction",
            "source.scalar.z-lens",
            "source.scalar.z-source",
            "source.vector.endpoint-log-redshift",
            "source.vector.impact-parameter-mpc",
            "source.vector.path-exposure",
            "source.vector.phi-nfw",
            "source.vector.phi-stars",
            "source.vector.signed-image-root",
        )
    )
)
_OUTPUTS = (
    "prediction.scalar.differential-log-redshift",
    "prediction.scalar.time-delay-days",
    "prediction.vector.endpoint-log-redshift",
    "prediction.vector.light-potential",
    "prediction.vector.matter-potential",
    "prediction.vector.path-exposure",
)
_RESPONSE_FOR = {value: value.replace("prediction.", "response.synthetic-", 1) for value in _OUTPUTS}
_AXES_FOR_OUTPUT = {
    "prediction.scalar.differential-log-redshift": ("pair",),
    "prediction.scalar.time-delay-days": ("pair",),
    "prediction.vector.endpoint-log-redshift": ("endpoint",),
    "prediction.vector.light-potential": ("image",),
    "prediction.vector.matter-potential": ("image",),
    "prediction.vector.path-exposure": ("image",),
}
_UNIT_FOR_OUTPUT = {
    "prediction.scalar.differential-log-redshift": "1",
    "prediction.scalar.time-delay-days": "day",
    "prediction.vector.endpoint-log-redshift": "1",
    "prediction.vector.light-potential": "1",
    "prediction.vector.matter-potential": "1",
    "prediction.vector.path-exposure": "1",
}
_LENS_SENSITIVE = tuple(value for value in _OUTPUTS if "endpoint-log-redshift" not in value)
_ENTRYPOINTS = {
    "ENDPOINT_LAPSE_CONTROL": "endpoint_lapse_control_adapter",
    "GEOMETRIC_TIME_DELAY_CONTROL": "geometric_time_delay_control_adapter",
    "GR_STARS_NFW_CONTROL": "gr_stars_nfw_control_adapter",
    "PATH_AGED_WEYL_CLOCK": "path_aged_weyl_clock_adapter",
    "PHI_PSI_SLIP_SAME_STATE": "phi_psi_slip_same_state_adapter",
}
_BLOCK_STATUS = {
    "AQUAL_SIMPLE_MU": BindingStatus.SOURCE_BLOCKED,
    "DPEL01_DISK_POLAR_ESCAPE_LOAD": BindingStatus.UNADAPTED,
    "MASHHOON_RAHVAR_NLG_Q0": BindingStatus.SOURCE_BLOCKED,
    "MICROLENSING_CHROMATIC_SHIFT": BindingStatus.SOURCE_BLOCKED,
    "MOVING_LENS_FREQUENCY_SHIFT": BindingStatus.SOURCE_BLOCKED,
    "QUMOND_SIMPLE_NU": BindingStatus.SOURCE_BLOCKED,
    "REFRACTED_GRAVITY_DISKMASS_MEDIAN": BindingStatus.SOURCE_BLOCKED,
}
_MPC_M = 3.085677581491367e22
_C_M_S = 299792458.0
_DAY_S = 86400.0
_SOLAR_MASS_KG = 1.98847e30


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
        raise SchemaViolation("lens synthetic path escaped repository")
    result = (_ROOT / parsed.as_posix()).resolve()
    if not result.is_relative_to(_ROOT):
        raise SchemaViolation("lens synthetic path escaped repository")
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
        "lenses",
        "mechanisms",
        "noise_families",
        "law_constants",
        "noise",
        "scoring",
        "geometry_mode",
        "time_mode",
        "coordinate_frame",
        "parameter_schema_path",
        "output_directory",
        "predecessor_binding",
        "source_anchors",
        "infrastructure_bindings",
        "adapter_blocks",
        "access_contract",
    }
    _require(set(config) == expected, "lens synthetic config keys changed")
    _require(
        config["schema"]
        == "open-gravity-lens-ray-clock-source-shaped-synthetic-injection-matrix-2.0",
        "lens synthetic schema changed",
    )
    _require(
        config["package_id"]
        == "open-gravity-lens-ray-clock-source-shaped-synthetic-injection-matrix-v2"
        and config["version"] == "v1.0.1",
        "lens synthetic identity changed",
    )
    _require(config["status"] == "FROZEN_SYNTHETIC_ONLY_PRE_RESPONSE", "status changed")
    _require(config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL", "claim changed")
    _require(
        config["experiment_id"] == "lens.ray-clock-source-shaped-synthetic.v1"
        and config["suite_seed"] == 26083103
        and config["geometry_mode"] == "two-image-point-lens-source-lift"
        and config["time_mode"] == "stationary-lens-path-clock"
        and config["coordinate_frame"] == "lens-centered-thin-lens"
        and config["parameter_schema_path"]
        == "configs/open_gravity_full3d_phangs_synthetic_adapter_empty_parameters_v1.schema.json",
        "lens synthetic scalar contract changed",
    )
    _require(
        len(config["lenses"]) == len(set(config["lenses"])) == 8
        and _json_sha256(config["lenses"])
        == "1aa65ce605cb47f1687ffcf1f43ed070f1e87c05ae59760d73a53792e473ce16",
        "lens set changed",
    )
    _require(config["mechanisms"] == sorted(_ENTRYPOINTS), "mechanisms changed")
    _require(
        config["noise_families"]
        == [
            "independent-source-envelope",
            "shared-mass-clock-nuisance",
            "zero-noise",
        ],
        "noise families changed",
    )
    _require(len(config["adapter_blocks"]) == 7, "adapter block count changed")
    _require(
        [row["formula_id"] for row in config["adapter_blocks"]] == sorted(_BLOCK_STATUS),
        "adapter block identities changed",
    )
    frozen_sections = {
        "law_constants": "6e2210fbc292e3f6bb4db4467827ebc7b8ef03bbbdc2ebb6e13628974d71bed6",
        "noise": "f07b981488183f445a9a8d674b629db192aa26969bd0e3db075402c2a79eea5b",
        "scoring": "a788bf5bf6fbfac36ad0463ce62572c75dcef29334fb46b293e2604f5b5e42ed",
        "source_anchors": "eadc0bab6d262a253e07aecf3eb30e48ec47d471f8024ef0325aa27cebc74124",
        "infrastructure_bindings": "1a3fdcf943e88ed23f81e246e4cb7b02168d00ead1114d7aefe43cebec23900c",
        "adapter_blocks": "c4f136fabe36f509b0ec531fda4f2bad6f37a2f0ad7d3deab2daaebb7bac227f",
        "access_contract": "906f26d371039f0f5bd0f8794a92bd962664a2ce310401f52b629417c259f3c2",
    }
    for key, expected_hash in frozen_sections.items():
        _require(_json_sha256(config[key]) == expected_hash, f"frozen section changed: {key}")
    _require(
        _repo_path(config["output_directory"]) == (_ROOT / OUTPUT_DIR).resolve(),
        "output path changed",
    )
    predecessor = config["predecessor_binding"]
    _require(
        _json_sha256(predecessor)
        == "235f7c5ddc5504db281d95eead079796c4fc925bcbec306d9eb981b8edcc0b02",
        "predecessor binding changed",
    )
    for prefix in ("config", "module", "test", "receipt"):
        path = _repo_path(predecessor[f"{prefix}_path"])
        if verify_hashes:
            _require(
                path.is_file()
                and _file_sha256(path) == predecessor[f"{prefix}_raw_sha256"],
                f"predecessor {prefix} drift",
            )
    if verify_hashes:
        predecessor_receipt = json.loads(
            _repo_path(predecessor["receipt_path"]).read_text(encoding="utf-8")
        )
        _require(
            predecessor_receipt["content_sha256"]
            == predecessor["receipt_content_sha256"],
            "predecessor receipt content drift",
        )
    access = config["access_contract"]
    _require(
        access["sealed_source_anchor_files_opened"] == 6
        and access["sealed_source_anchor_bytes_opened"] == 113366,
        "source anchor ceiling changed",
    )
    _require(
        all(
            access[key] == 0
            for key in (
                "eso_pixels_decoded",
                "eso_spectral_rows_decoded",
                "slacs_confirmation_files_opened",
                "lens_response_tables_opened",
                "lens_response_rows_opened",
                "theory_or_nuisance_tuning_events",
                "network_calls",
                "model_calls",
                "paid_calls",
            )
        ),
        "response access boundary changed",
    )
    for group in (config["source_anchors"], config["infrastructure_bindings"]):
        ids = [row["id"] for row in group]
        _require(ids == sorted(set(ids)), "binding IDs must be sorted unique")
        for row in group:
            path = _repo_path(row["path"])
            if verify_hashes:
                _require(path.is_file() and _file_sha256(path) == row["sha256"], f"binding drift: {row['id']}")
                if "bytes" in row:
                    _require(path.stat().st_size == row["bytes"], f"binding size drift: {row['id']}")


def _binding_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    rows = (*config["source_anchors"], *config["infrastructure_bindings"])
    return {row["id"]: _file_sha256(_repo_path(row["path"])) for row in rows}


def _catalogue(config: Mapping[str, Any]):
    provenance = canonical_sha256(config["source_anchors"])
    specs: list[tuple[str, str, tuple[str, ...], int, str]] = [
        ("source.scalar.einstein-mass-kg", "lens Einstein mass", ("object",), 0, "kg"),
        ("source.scalar.geometric-time-delay-days", "geometric image delay", ("pair",), 0, "day"),
        ("source.scalar.gr-time-delay-days", "GR point-lens image delay", ("pair",), 0, "day"),
        ("source.scalar.gravitational-radius-mpc", "model gravitational radius", ("object",), 0, "Mpc"),
        ("source.scalar.image-time-delay-available", "published delay availability", ("object",), 0, "integer code"),
        ("source.scalar.image-time-delay-days", "published image time delay", ("pair",), 0, "day"),
        ("source.scalar.image-time-delay-uncertainty-days", "published delay uncertainty", ("pair",), 0, "day"),
        ("source.scalar.lens-code", "lens identity code", ("object",), 0, "integer code"),
        ("source.scalar.path-age-mpc", "source-to-lens baryon-frame path age", ("object",), 0, "Mpc"),
        ("source.scalar.stellar-mass-fraction", "frozen stars fraction", ("object",), 0, "1"),
        ("source.scalar.z-lens", "lens redshift", ("object",), 0, "1"),
        ("source.scalar.z-source", "source redshift", ("object",), 0, "1"),
        ("source.vector.endpoint-log-redshift", "endpoint ln(1+z)", ("endpoint",), 0, "1"),
        ("source.vector.impact-parameter-mpc", "two-image impact geometry", ("image",), 0, "Mpc"),
        ("source.vector.path-exposure", "two-image Weyl path exposure", ("image",), 0, "1"),
        ("source.vector.phi-nfw", "source-shaped NFW potential", ("image",), 0, "1"),
        ("source.vector.phi-stars", "source-shaped stellar potential", ("image",), 0, "1"),
        ("source.vector.signed-image-root", "signed point-lens image root", ("image",), 0, "1"),
    ]
    for output in _OUTPUTS:
        specs.append((output, output, _AXES_FOR_OUTPUT[output], 0, _UNIT_FOR_OUTPUT[output]))
        specs.append((_RESPONSE_FOR[output], _RESPONSE_FOR[output], _AXES_FOR_OUTPUT[output], 0, _UNIT_FOR_OUTPUT[output]))
    specs.append(
        (
            "truth.scalar.injection-id",
            "synthetic mechanism identity",
            ("object",),
            0,
            "integer code",
        )
    )
    dimensions = {
        "1": (0, 0, 0, 0, 0, 0, 0),
        "integer code": (0, 0, 0, 0, 0, 0, 0),
        "kg": (1, 0, 0, 0, 0, 0, 0),
        "Mpc": (0, 1, 0, 0, 0, 0, 0),
        "day": (0, 0, 1, 0, 0, 0, 0),
    }
    elements = []
    for element_id, quantity, axes, rank, unit in specs:
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
                frame="latent" if element_id.startswith("truth.") else config["coordinate_frame"],
                support="two-image response-blind Lane1/Lane7 source lift",
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
    return catalogue_from_elements("open-gravity-lens-ray-clock-source-shaped-synthetic", "v1.0.0", elements)


def _bindings(config: Mapping[str, Any]) -> tuple[FormulaExecutionBinding, ...]:
    hashes = _binding_hashes(config)
    schema_sha = _file_sha256(_repo_path(config["parameter_schema_path"]))
    block_reason = {row["formula_id"]: row["reason"] for row in config["adapter_blocks"]}
    rows = []
    for formula_id in sorted((*_ENTRYPOINTS, *_BLOCK_STATUS)):
        executable = formula_id in _ENTRYPOINTS
        rows.append(
            FormulaExecutionBinding(
                binding_id=f"binding.lens-ray-clock.{formula_id.lower()}.v1",
                formula_id=formula_id,
                formula_version="v1.0.0-source-shaped-synthetic",
                formula_sha256=canonical_sha256(
                    {
                        "formula_id": formula_id,
                        "lane1_module_sha256": hashes["LANE1_MODULE"],
                        "source_contract": "two-image-same-state-phi-psi-v1",
                        "blocked_reason": block_reason.get(formula_id),
                    }
                ),
                status=BindingStatus.EXECUTABLE if executable else _BLOCK_STATUS[formula_id],
                entrypoint=(
                    f"sigma_theory_compiler.open_gravity_lens_ray_clock_source_shaped_synthetic_injection_matrix_v2:{_ENTRYPOINTS[formula_id]}"
                    if executable
                    else None
                ),
                required_features=_FEATURES if executable else ("source.scalar.mass-density",),
                optional_features=(),
                emitted_features=_OUTPUTS,
                domains=(("strong-lens",) if executable else ("cluster", "disk-galaxy")),
                geometry_support=((config["geometry_mode"],) if executable else ("density-grid",)),
                time_support=((config["time_mode"],) if executable else ("static",)),
                parameter_schema_path=config["parameter_schema_path"],
                parameter_schema_sha256=schema_sha,
                approximation_ceiling=(
                    "synthetic-only two-image point-lens source lift; no pixels, spectra, response table, or fitted lens mass model"
                    if executable
                    else block_reason[formula_id]
                ),
                health_gates=("determinism", "finite-output", "limit", "source-hash", "typed-output", "unit"),
                resource_bounds=ResourceBounds(30, 500_000_000, 2_000_000),
            )
        )
    return tuple(rows)


def _prediction(features: Mapping[str, Any], mode: str, constants: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
    _require(set(features) == set(_FEATURES), "lens feature projection changed")
    config = load_config()
    law = dict(config["law_constants"] if constants is None else constants)
    phi = np.asarray(features["source.vector.phi-stars"], dtype=np.float64) + np.asarray(
        features["source.vector.phi-nfw"], dtype=np.float64
    )
    exposure = np.asarray(features["source.vector.path-exposure"], dtype=np.float64)
    endpoint = np.asarray(features["source.vector.endpoint-log-redshift"], dtype=np.float64)
    gr_delay = np.asarray(features["source.scalar.gr-time-delay-days"], dtype=np.float64)
    geometric_delay = np.asarray(
        features["source.scalar.geometric-time-delay-days"], dtype=np.float64
    )
    light = 2.0 * phi
    delay = np.array(gr_delay, copy=True)
    output_exposure = np.array(exposure, copy=True)
    delta = 0.0
    if mode == "endpoint":
        endpoint = endpoint + np.asarray([float(np.mean(phi)), 0.0], dtype=np.float64)
    elif mode == "geometric":
        delay = np.array(geometric_delay, copy=True)
    elif mode == "path":
        delta = float(law["path_alpha"]) * float(exposure[0] - exposure[1])
    elif mode == "slip":
        gamma = float(law["same_state_slip_gamma"])
        scale = (1.0 + gamma) / 2.0
        light = (1.0 + gamma) * phi
        delay = gr_delay * scale
        output_exposure = exposure * math.sqrt(scale)
        delta = (
            float(law["slip_path_fraction"])
            * (gamma - 1.0)
            * float(exposure[0] - exposure[1])
        )
    else:
        _require(mode == "gr", "unknown lens mechanism")
    result = {
        "prediction.scalar.differential-log-redshift": np.asarray([delta], dtype=np.float64),
        "prediction.scalar.time-delay-days": np.asarray(delay, dtype=np.float64),
        "prediction.vector.endpoint-log-redshift": np.asarray(endpoint, dtype=np.float64),
        "prediction.vector.light-potential": np.asarray(light, dtype=np.float64),
        "prediction.vector.matter-potential": np.asarray(phi, dtype=np.float64),
        "prediction.vector.path-exposure": np.asarray(output_exposure, dtype=np.float64),
    }
    _require(all(np.all(np.isfinite(value)) for value in result.values()), "nonfinite lens prediction")
    return result


def endpoint_lapse_control_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "endpoint control has no free parameters")
    return _prediction(features, "endpoint")


def geometric_time_delay_control_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "geometric delay control has no free parameters")
    return _prediction(features, "geometric")


def gr_stars_nfw_control_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "GR control has no free parameters")
    return _prediction(features, "gr")


def path_aged_weyl_clock_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "path clock has no free parameters")
    return _prediction(features, "path")


def phi_psi_slip_same_state_adapter(features: Mapping[str, Any], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(not parameters, "Phi/Psi slip has no free parameters")
    return _prediction(features, "slip")


def _delay_value(row: Mapping[str, Any]) -> tuple[float, float, bool]:
    if "published_delta_t_ab_days" in row:
        delay = float(row["published_delta_t_ab_days"])
    elif "published_delta_t_days" in row:
        delay = float(row["published_delta_t_days"])
    else:
        return 0.0, 0.0, False
    raw = row.get("uncertainty_days", 0.0)
    if isinstance(raw, Mapping):
        uncertainty = 0.5 * (abs(float(raw["minus"])) + abs(float(raw["plus"])))
    else:
        uncertainty = abs(float(raw))
    return delay, uncertainty, True


def _source_items(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    anchors = {row["id"]: row for row in config["source_anchors"]}
    lane1_receipt = json.loads(_repo_path(anchors["LANE1_RECEIPT"]["path"]).read_text(encoding="utf-8"))
    lane7_receipt = json.loads(_repo_path(anchors["LANE7_RECEIPT"]["path"]).read_text(encoding="utf-8"))
    ledger = json.loads(_repo_path(anchors["LANE7_LEDGER"]["path"]).read_text(encoding="utf-8"))
    _require(lane1_receipt["content_sha256"] == _json_sha256({k: v for k, v in lane1_receipt.items() if k != "content_sha256"}), "Lane1 receipt self hash")
    _require(lane7_receipt["content_sha256"] == _json_sha256({k: v for k, v in lane7_receipt.items() if k != "content_sha256"}), "Lane7 receipt self hash")
    _require(ledger["content_sha256"] == _json_sha256({k: v for k, v in ledger.items() if k != "content_sha256"}), "Lane7 ledger self hash")
    _require(lane1_receipt["status"] == "PASS_REPAIRED_COVARIANT_KINEMATIC_PATH_LAW_AND_RESPONSE_BLIND_PREFLIGHT", "Lane1 status")
    _require(lane7_receipt["status"] == "SEALED_SOURCE_PREFLIGHT_ALL_EIGHT_RESPONSE_BLOCKED_ONE_PHASE_ALIGNED_PARTIAL", "Lane7 status")
    _require(ledger["lens_count"] == 8 and ledger["access_accounting"]["response_scores_computed"] == 0, "Lane7 response boundary")
    ledger_by_name = {row["name"]: row for row in ledger["lenses"]}
    with _repo_path(anchors["LANE1_PREDICTIONS"]["path"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    _require(len(source_rows) == 8, "Lane1 exploration count")
    by_name = {row["name"]: row for row in source_rows}
    _require(list(by_name) == config["lenses"], "Lane1 lens order changed")
    results = []
    for lens_code, name in enumerate(config["lenses"]):
        row = by_name[name]
        source_ledger = ledger_by_name[name]
        _require(row["response_opened"] == "False" and row["confirmation_predictor_used"] == "False", "response source leaked")
        _require(source_ledger["source_status"].startswith("SOURCE_"), "Lane7 lens status")
        inner = float(row["inner_impact_mpc"])
        outer = float(row["outer_impact_mpc"])
        roots = np.asarray([float(row["signed_x_minus"]), float(row["signed_x_plus"])], dtype=np.float64)
        impacts = np.asarray([inner, outer], dtype=np.float64)
        exposure = np.asarray([float(row["inner_exposure"]), float(row["outer_exposure"])], dtype=np.float64)
        r_g = float(row["model_gravitational_radius_mpc"])
        einstein_radius = math.sqrt(inner * outer)
        flux_ratio = float(row["image_flux_ratio"])
        stellar_fraction = float(np.clip(0.55 + 0.1 * (flux_ratio - 0.5), 0.45, 0.65))
        core = float(config["law_constants"]["stellar_core_over_einstein_radius"]) * einstein_radius
        nfw_scale = float(config["law_constants"]["nfw_scale_over_einstein_radius"]) * einstein_radius
        phi_stars = -stellar_fraction * r_g / einstein_radius * math.sqrt(einstein_radius**2 + core**2) / np.sqrt(impacts**2 + core**2)
        nfw_norm = math.log1p(einstein_radius / nfw_scale)
        phi_nfw = -(1.0 - stellar_fraction) * r_g / impacts * np.log1p(impacts / nfw_scale) / nfw_norm
        y = float(row["source_offset_over_einstein_radius"])
        tau = 0.5 * (roots - y) ** 2 - np.log(np.abs(roots))
        tau_geometry = 0.5 * (roots - y) ** 2
        factor_days = (1.0 + float(row["z_lens"])) * 4.0 * r_g * _MPC_M / (_C_M_S * _DAY_S)
        gr_delay = factor_days * float(tau[0] - tau[1])
        geometric_delay = factor_days * float(tau_geometry[0] - tau_geometry[1])
        published_delay, published_uncertainty, available = _delay_value(source_ledger["time_delay"])
        values = {
            "source.scalar.einstein-mass-kg": np.asarray([float(row["model_einstein_mass_msun"]) * _SOLAR_MASS_KG], dtype=np.float64),
            "source.scalar.geometric-time-delay-days": np.asarray([geometric_delay], dtype=np.float64),
            "source.scalar.gr-time-delay-days": np.asarray([gr_delay], dtype=np.float64),
            "source.scalar.gravitational-radius-mpc": np.asarray([r_g], dtype=np.float64),
            "source.scalar.image-time-delay-available": np.asarray([int(available)], dtype=np.int64),
            "source.scalar.image-time-delay-days": np.asarray([published_delay], dtype=np.float64),
            "source.scalar.image-time-delay-uncertainty-days": np.asarray([published_uncertainty], dtype=np.float64),
            "source.scalar.lens-code": np.asarray([lens_code], dtype=np.int64),
            "source.scalar.path-age-mpc": np.asarray([float(row["source_to_lens_path_mpc"])], dtype=np.float64),
            "source.scalar.stellar-mass-fraction": np.asarray([stellar_fraction], dtype=np.float64),
            "source.scalar.z-lens": np.asarray([float(row["z_lens"])], dtype=np.float64),
            "source.scalar.z-source": np.asarray([float(row["z_source"])], dtype=np.float64),
            "source.vector.endpoint-log-redshift": np.log1p(np.asarray([float(row["z_lens"]), float(row["z_source"])], dtype=np.float64)),
            "source.vector.impact-parameter-mpc": impacts,
            "source.vector.path-exposure": exposure,
            "source.vector.phi-nfw": np.asarray(phi_nfw, dtype=np.float64),
            "source.vector.phi-stars": np.asarray(phi_stars, dtype=np.float64),
            "source.vector.signed-image-root": roots,
        }
        _require(all(np.all(np.isfinite(value)) for value in values.values()), "nonfinite source lift")
        results.append(
            {
                "lens": name,
                "values": values,
                "metadata": {
                    "published_delay_available": available,
                    "published_delay_days": published_delay,
                    "published_delay_uncertainty_days": published_uncertainty,
                    "source_status": source_ledger["source_status"],
                    "maximum_lens_equation_residual": float(row["maximum_lens_equation_residual"]),
                    "mass_closure_relative_error": float(row["mass_closure_relative_error"]),
                    "path_measure_quadrature_relative_error": float(row["path_measure_quadrature_relative_error"]),
                    "flux_ratio_reconstruction_absolute_error": float(row["flux_ratio_reconstruction_absolute_error"]),
                    "separation_reconstruction_absolute_error_arcsec": float(row["separation_reconstruction_absolute_error_arcsec"]),
                    "source_feature_sha256": {key: array_sha256(value) for key, value in values.items()},
                },
            }
        )
    return results


def _scenario(
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    scenario_id: str,
    truth_world_id: str,
    nuisance_draw: int,
    responses: Mapping[str, np.ndarray],
    variances: Mapping[str, np.ndarray],
    truth_index: int,
) -> ScenarioDescriptor:
    values = item["values"]
    feature_axes = {
        key: (
            ("image",)
            if key.startswith("source.vector.") and "endpoint" not in key
            else ("endpoint",)
            if "endpoint-log-redshift" in key
            else ("pair",)
            if "time-delay" in key and "available" not in key
            else ("object",)
        )
        for key in _FEATURES
    }
    feature_units = {
        key: (
            "kg"
            if "mass-kg" in key
            else "Mpc"
            if "mpc" in key
            else "day"
            if "time-delay" in key and "available" not in key
            else "integer code"
            if key.endswith(("available", "lens-code"))
            else "1"
        )
        for key in _FEATURES
    }
    anchors = {row["id"]: row for row in config["source_anchors"]}
    truth = np.asarray([truth_index], dtype=np.int64)
    return ScenarioDescriptor(
        scenario_id=scenario_id,
        object_id=_array_key(str(item["lens"])),
        experiment_id=config["experiment_id"],
        domain="strong-lens",
        geometry_mode=config["geometry_mode"],
        time_mode=config["time_mode"],
        coordinate_frame=config["coordinate_frame"],
        axes=(
            AxisSpec("endpoint", 2, None, None),
            AxisSpec("image", 2, None, None),
            AxisSpec("object", 1, None, None),
            AxisSpec("pair", 1, None, None),
        ),
        formula_features=tuple(
            FeatureValueRef(
                key,
                VALUES_PATH.as_posix(),
                array_sha256(values[key]),
                values[key].dtype.name,
                values[key].shape,
                feature_axes[key],
                feature_units[key],
                config["coordinate_frame"],
            )
            for key in _FEATURES
        ),
        scoring_responses=tuple(
            FeatureValueRef(
                _RESPONSE_FOR[key],
                VALUES_PATH.as_posix(),
                array_sha256(responses[key]),
                "float64",
                responses[key].shape,
                _AXES_FOR_OUTPUT[key],
                _UNIT_FOR_OUTPUT[key],
                config["coordinate_frame"],
            )
            for key in _OUTPUTS
        ),
        hidden_truth=(
            FeatureValueRef(
                "truth.scalar.injection-id",
                VALUES_PATH.as_posix(),
                array_sha256(truth),
                "int64",
                truth.shape,
                ("object",),
                "integer code",
                "latent",
            ),
        ),
        expected_predictions=tuple(
            EmittedPredictionSpec(
                key,
                VALUES_PATH.as_posix(),
                "float64",
                responses[key].shape,
                _AXES_FOR_OUTPUT[key],
                _UNIT_FOR_OUTPUT[key],
                config["coordinate_frame"],
            )
            for key in _OUTPUTS
        ),
        uncertainties=tuple(
            UncertaintyRef(
                f"uncertainty.{_RESPONSE_FOR[key]}",
                _RESPONSE_FOR[key],
                "diagonal-covariance",
                VALUES_PATH.as_posix(),
                array_sha256(variances[key]),
            )
            for key in _OUTPUTS
        ),
        anchors=(
            AnchorBinding("anchor.lane1-response-blind-predictions", anchors["LANE1_PREDICTIONS"]["path"], anchors["LANE1_PREDICTIONS"]["sha256"]),
            AnchorBinding("anchor.lane7-source-only-ledger", anchors["LANE7_LEDGER"]["path"], anchors["LANE7_LEDGER"]["sha256"]),
        ),
        seed_lineage=SeedLineage(
            config["suite_seed"],
            scenario_id,
            _array_key(str(item["lens"])),
            truth_world_id,
            nuisance_draw,
            0,
        ),
    )


def _noise_response(
    truth: Mapping[str, np.ndarray],
    family: str,
    lineage: SeedLineage,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(lineage.derived_seed)
    sigmas = {}
    for key in _OUTPUTS:
        value = np.asarray(truth[key], dtype=np.float64)
        rms = max(float(np.sqrt(np.mean(value * value))), np.finfo(float).tiny)
        floor = max(
            float(config["noise"]["floor_fraction_of_feature_rms"]) * rms,
            float(config["noise"]["absolute_sigma_by_output"][key]),
        )
        sigmas[key] = np.maximum(float(config["noise"]["fractional_sigma"]) * np.abs(value), floor)
    mass_scale = 1.0
    clock_offset = 0.0
    if family == "shared-mass-clock-nuisance":
        mass_scale += float(config["noise"]["shared_mass_scale_sigma"]) * float(rng.normal())
        clock_offset = float(config["noise"]["shared_clock_offset_sigma"]) * float(rng.normal())
    responses = {}
    variances = {}
    for key in _OUTPUTS:
        value = np.asarray(truth[key], dtype=np.float64)
        if family == "zero-noise":
            response = np.array(value, copy=True)
        elif family == "independent-source-envelope":
            response = value + rng.normal(size=value.shape) * sigmas[key]
        else:
            _require(family == "shared-mass-clock-nuisance", "unknown lens noise family")
            baseline = (
                value + clock_offset
                if key == "prediction.vector.endpoint-log-redshift"
                else mass_scale * value
            )
            response = baseline + rng.normal(size=value.shape) * sigmas[key] * float(
                config["noise"]["shared_residual_fraction"]
            )
        responses[key] = np.asarray(response, dtype=np.float64)
        variances[key] = np.square(sigmas[key], dtype=np.float64)
        _require(np.all(variances[key] > 0.0), "invalid synthetic variance")
    return responses, variances, {
        "family": family,
        "derived_seed": lineage.derived_seed,
        "mass_scale": mass_scale,
        "endpoint_clock_offset": clock_offset,
        "real_response_used": False,
    }


def _profiled_metrics(
    candidate: Mapping[str, np.ndarray],
    response: Mapping[str, np.ndarray],
    variance: Mapping[str, np.ndarray],
) -> dict[str, float]:
    lens_x = np.concatenate([np.asarray(candidate[key]).reshape(-1) for key in _LENS_SENSITIVE])
    lens_y = np.concatenate([np.asarray(response[key]).reshape(-1) for key in _LENS_SENSITIVE])
    lens_v = np.concatenate([np.asarray(variance[key]).reshape(-1) for key in _LENS_SENSITIVE])
    weight = 1.0 / lens_v
    denominator = float(np.dot(weight * lens_x, lens_x))
    _require(math.isfinite(denominator) and denominator > 0.0, "invalid lens amplitude nuisance")
    scale = float(np.dot(weight * lens_x, lens_y) / denominator)
    endpoint_key = "prediction.vector.endpoint-log-redshift"
    endpoint_x = np.asarray(candidate[endpoint_key])
    endpoint_y = np.asarray(response[endpoint_key])
    endpoint_v = np.asarray(variance[endpoint_key])
    endpoint_weight = 1.0 / endpoint_v
    clock_offset = float(np.sum(endpoint_weight * (endpoint_y - endpoint_x)) / np.sum(endpoint_weight))
    fitted = {
        key: (
            np.asarray(candidate[key]) + clock_offset
            if key == endpoint_key
            else scale * np.asarray(candidate[key])
        )
        for key in _OUTPUTS
    }
    whitened = []
    relative = []
    for key in _OUTPUTS:
        residual = fitted[key] - np.asarray(response[key])
        whitened.extend((residual * residual / np.asarray(variance[key])).reshape(-1))
        relative.append(
            float(
                np.linalg.norm(residual)
                / max(np.linalg.norm(np.asarray(response[key])), np.finfo(float).tiny)
            )
        )
    return {
        "profiled_whitened_rmse": float(math.sqrt(float(np.mean(whitened)))),
        "mean_feature_relative_rmse": float(np.mean(relative)),
        "fitted_lens_amplitude_scale": scale,
        "fitted_endpoint_clock_offset": clock_offset,
    }


def _array_key(*parts: str) -> str:
    return "__".join(
        part.lower().replace("-", "_").replace(".", "_").replace(" ", "_").replace("+", "p")
        for part in parts
    )


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: arrays[key] for key in sorted(arrays)})
    return buffer.getvalue()


def _diagnostics(
    config: Mapping[str, Any],
    source_items: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    geometry_rows = []
    limit_rows = []
    pair_rows = []
    for item in source_items:
        lens = str(item["lens"])
        metadata = item["metadata"]
        values = item["values"]
        gr = predictions[(lens, "GR_STARS_NFW_CONTROL")]
        _require(np.array_equal(gr["prediction.vector.light-potential"], 2.0 * gr["prediction.vector.matter-potential"]), "GR same-state Phi/Psi")
        alpha_zero = dict(config["law_constants"])
        alpha_zero["path_alpha"] = 0.0
        path_zero = _prediction(values, "path", alpha_zero)
        gamma_one = dict(config["law_constants"])
        gamma_one["same_state_slip_gamma"] = 1.0
        slip_gr = _prediction(values, "slip", gamma_one)
        equal_values = {key: np.array(value, copy=True) for key, value in values.items()}
        equal_values["source.vector.path-exposure"][:] = float(np.mean(values["source.vector.path-exposure"]))
        equal_delta = float(_prediction(equal_values, "path")["prediction.scalar.differential-log-redshift"][0])
        swapped = {key: np.array(value, copy=True) for key, value in values.items()}
        swapped["source.vector.path-exposure"] = swapped["source.vector.path-exposure"][::-1]
        path = predictions[(lens, "PATH_AGED_WEYL_CLOCK")]
        swapped_delta = float(_prediction(swapped, "path")["prediction.scalar.differential-log-redshift"][0])
        zero = {key: np.array(value, copy=True) for key, value in values.items()}
        for key in (
            "source.scalar.geometric-time-delay-days",
            "source.scalar.gr-time-delay-days",
            "source.scalar.gravitational-radius-mpc",
            "source.vector.path-exposure",
            "source.vector.phi-nfw",
            "source.vector.phi-stars",
        ):
            zero[key][...] = 0
        zero_prediction = _prediction(zero, "path")
        zero_lens_max = max(
            float(np.max(np.abs(zero_prediction[key]))) for key in _LENS_SENSITIVE
        )
        geometry_rows.append(
            {
                "lens": lens,
                **{
                    key: metadata[key]
                    for key in (
                        "maximum_lens_equation_residual",
                        "mass_closure_relative_error",
                        "path_measure_quadrature_relative_error",
                        "flux_ratio_reconstruction_absolute_error",
                        "separation_reconstruction_absolute_error_arcsec",
                    )
                },
            }
        )
        limit_rows.append(
            {
                "lens": lens,
                "alpha_zero_max_abs_error": max(float(np.max(np.abs(path_zero[key] - gr[key]))) for key in _OUTPUTS),
                "gamma_one_max_abs_error": max(float(np.max(np.abs(slip_gr[key] - gr[key]))) for key in _OUTPUTS),
                "equal_exposure_differential": equal_delta,
                "image_swap_antisymmetry_error": abs(swapped_delta + float(path["prediction.scalar.differential-log-redshift"][0])),
                "zero_mass_lens_signal_max": zero_lens_max,
                "gr_phi_psi_same_state_error": float(np.max(np.abs(gr["prediction.vector.light-potential"] - 2.0 * gr["prediction.vector.matter-potential"]))),
            }
        )
        for left_index, left in enumerate(config["mechanisms"]):
            for right in config["mechanisms"][left_index + 1 :]:
                left_prediction = predictions[(lens, left)]
                right_prediction = predictions[(lens, right)]
                variances = {
                    key: np.square(
                        np.maximum(
                            float(config["noise"]["fractional_sigma"]) * np.abs(left_prediction[key]),
                            max(
                                float(config["noise"]["floor_fraction_of_feature_rms"])
                                * max(
                                    float(np.sqrt(np.mean(left_prediction[key] ** 2))),
                                    np.finfo(float).tiny,
                                ),
                                float(config["noise"]["absolute_sigma_by_output"][key]),
                            ),
                        )
                    )
                    for key in _OUTPUTS
                }
                metric = _profiled_metrics(right_prediction, left_prediction, variances)
                pair_rows.append(
                    {
                        "lens": lens,
                        "left_formula_id": left,
                        "right_formula_id": right,
                        "profiled_whitened_rmse": metric["profiled_whitened_rmse"],
                        "degenerate_by_frozen_threshold": metric["profiled_whitened_rmse"]
                        <= float(config["scoring"]["pairwise_degeneracy_profiled_rmse_max"]),
                    }
                )
    maximums = {
        "maximum_lens_equation_residual": max(row["maximum_lens_equation_residual"] for row in geometry_rows),
        "maximum_mass_closure_relative_error": max(row["mass_closure_relative_error"] for row in geometry_rows),
        "maximum_path_quadrature_relative_error": max(row["path_measure_quadrature_relative_error"] for row in geometry_rows),
        "maximum_flux_ratio_reconstruction_error": max(row["flux_ratio_reconstruction_absolute_error"] for row in geometry_rows),
        "maximum_separation_reconstruction_error_arcsec": max(row["separation_reconstruction_absolute_error_arcsec"] for row in geometry_rows),
        "maximum_alpha_zero_error": max(row["alpha_zero_max_abs_error"] for row in limit_rows),
        "maximum_gamma_one_error": max(row["gamma_one_max_abs_error"] for row in limit_rows),
        "maximum_equal_exposure_differential": max(abs(row["equal_exposure_differential"]) for row in limit_rows),
        "maximum_image_swap_antisymmetry_error": max(row["image_swap_antisymmetry_error"] for row in limit_rows),
        "maximum_zero_mass_lens_signal": max(row["zero_mass_lens_signal_max"] for row in limit_rows),
        "maximum_gr_phi_psi_same_state_error": max(row["gr_phi_psi_same_state_error"] for row in limit_rows),
    }
    return {
        "schema": "open-gravity-lens-ray-clock-invariance-identifiability-1.0",
        "geometry_rows": geometry_rows,
        "limit_rows": limit_rows,
        "pairwise_identifiability": pair_rows,
        "pair_count": len(pair_rows),
        "degenerate_pair_count": sum(row["degenerate_by_frozen_threshold"] for row in pair_rows),
        **maximums,
        "pass": all(value <= 1.0e-12 for value in maximums.values()),
    }


def derive_release() -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, bytes]:
    config = load_config()
    validate_config(config)
    source_items = _source_items(config)
    catalogue = _catalogue(config)
    bindings = _bindings(config)
    validate_binding_catalogue(bindings, catalogue)
    executable = tuple(row for row in bindings if row.status is BindingStatus.EXECUTABLE)
    registrations = tuple(
        AdapterRegistration.create(f"adapter.lens-ray-clock.{row.formula_id.lower()}.v1", row)
        for row in executable
    )
    validate_adapter_registry(registrations)
    registration_by_formula = {row.formula_binding.formula_id: row for row in registrations}
    module_sha = _file_sha256(Path(__file__))
    anchor_hashes = _binding_hashes(config)
    release = SyntheticSuiteRelease(
        suite_id=config["package_id"],
        version=config["version"],
        release_sha256=canonical_sha256(
            {
                "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
                "generator_raw_sha256": module_sha,
                "lane1_receipt_sha256": anchor_hashes["LANE1_RECEIPT"],
                "lane7_receipt_sha256": anchor_hashes["LANE7_RECEIPT"],
            }
        ),
        ontology_sha256=catalogue.content_sha256,
        generator_sha256=module_sha,
        observation_operator_sha256=_json_sha256(
            {"noise": config["noise"], "noise_families": config["noise_families"], "scoring": config["scoring"]}
        ),
        changed_feature_ids=(*_OUTPUTS, *(_RESPONSE_FOR[value] for value in _OUTPUTS), "truth.scalar.injection-id"),
        change_level="MAJOR",
        response_calibrated=False,
        prediction_semantics_changed=True,
    )
    arrays: dict[str, np.ndarray] = {}
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    predictions: dict[tuple[str, str], Mapping[str, np.ndarray]] = {}
    for item in source_items:
        lens = str(item["lens"])
        for key, value in item["values"].items():
            arrays[_array_key("source", lens, key)] = value
        zero_response = {key: np.zeros(_AXES_FOR_OUTPUT[key] == ("image",) and 2 or _AXES_FOR_OUTPUT[key] == ("endpoint",) and 2 or 1, dtype=np.float64) for key in _OUTPUTS}
        zero_variance = {key: np.ones_like(value) for key, value in zero_response.items()}
        scaffold = _scenario(
            config,
            item,
            f"lens.{_array_key(lens)}.execution-scaffold.v1",
            "truth.execution-scaffold",
            0,
            zero_response,
            zero_variance,
            0,
        )
        validate_scenario_catalogue(scaffold, catalogue)
        for binding in executable:
            decision = decide_scenario_eligibility(binding, catalogue, scaffold)
            _require(decision.status is EligibilityStatus.ELIGIBLE, "executable lens binding ineligible")
            result = execute_binding_in_process(
                binding,
                catalogue,
                scaffold,
                {key: item["values"][key] for key in binding.required_features},
                {},
            )
            output = {key: np.asarray(result.output_values[key], dtype=np.float64) for key in _OUTPUTS}
            value_keys = {}
            value_hashes = {}
            for key, value in output.items():
                value_key = _array_key("candidate", lens, binding.formula_id, key)
                arrays[value_key] = value
                value_keys[key] = value_key
                value_hashes[key] = array_sha256(value)
            cache[(lens, binding.formula_id)] = {
                "prediction": output,
                "value_keys": value_keys,
                "value_sha256": value_hashes,
                "output_sha256": result.output_sha256,
                "scaffold_scenario_sha256": scaffold.content_sha256,
            }
            predictions[(lens, binding.formula_id)] = output
    diagnostics = _diagnostics(config, source_items, predictions)
    _require(diagnostics["pass"], "lens invariant/unit/limit gates failed")
    diagnostics_bytes = _json_bytes(diagnostics, indent=2)
    ledger = SyntheticReplayLedger("gravity.synthetic.lens-ray-clock-source-shaped-matrix.v2", ())
    scenario_rows = []
    confusion_counts = {
        truth: {candidate.formula_id: 0 for candidate in executable}
        for truth in config["mechanisms"]
    }
    recovery_by_truth = {
        truth: {"scenarios": 0, "recovered": 0, "distinct": 0}
        for truth in config["mechanisms"]
    }
    comparison_count = 0
    blocked_count = 0
    recovered_count = 0
    distinct_count = 0
    for item in source_items:
        lens = str(item["lens"])
        for truth_index, truth_formula in enumerate(config["mechanisms"]):
            truth_prediction = predictions[(lens, truth_formula)]
            for nuisance_draw, family in enumerate(config["noise_families"]):
                truth_world_id = f"truth.{truth_formula.lower()}"
                scenario_id = f"lens.{_array_key(lens)}.{truth_world_id}.noise-{family}.v1"
                lineage = SeedLineage(
                    config["suite_seed"],
                    scenario_id,
                    _array_key(lens),
                    truth_world_id,
                    nuisance_draw,
                    0,
                )
                responses, variances, noise = _noise_response(
                    truth_prediction, family, lineage, config
                )
                scenario = _scenario(
                    config,
                    item,
                    scenario_id,
                    truth_world_id,
                    nuisance_draw,
                    responses,
                    variances,
                    truth_index,
                )
                truth_value = np.asarray([truth_index], dtype=np.int64)
                validate_scenario_catalogue(scenario, catalogue)
                validate_scenario_values(
                    scenario,
                    formula_values=item["values"],
                    response_values={_RESPONSE_FOR[key]: responses[key] for key in _OUTPUTS},
                    truth_values={"truth.scalar.injection-id": truth_value},
                    uncertainty_values={f"uncertainty.{_RESPONSE_FOR[key]}": variances[key] for key in _OUTPUTS},
                )
                response_locators = {}
                variance_locators = {}
                for key in _OUTPUTS:
                    response_key = _array_key("response", lens, truth_formula, family, key)
                    variance_key = _array_key("variance", lens, truth_formula, family, key)
                    arrays[response_key] = responses[key]
                    arrays[variance_key] = variances[key]
                    response_locators[key] = {"key": response_key, "sha256": array_sha256(responses[key])}
                    variance_locators[key] = {"key": variance_key, "sha256": array_sha256(variances[key])}
                truth_key = _array_key("truth", lens, truth_formula, family)
                arrays[truth_key] = truth_value
                comparisons = []
                for binding in executable:
                    cached = cache[(lens, binding.formula_id)]
                    metric = _profiled_metrics(cached["prediction"], responses, variances)
                    comparison_count += 1
                    comparisons.append(
                        {
                            "candidate_formula_id": binding.formula_id,
                            "binding_sha256": binding.content_sha256,
                            "adapter_sha256": registration_by_formula[binding.formula_id].adapter_sha256,
                            "metrics": metric,
                            "numerical_valid": True,
                            "output_sha256": cached["output_sha256"],
                            "source_cache_scenario_sha256": cached["scaffold_scenario_sha256"],
                            "value_keys": cached["value_keys"],
                            "value_sha256": cached["value_sha256"],
                        }
                    )
                ordered = sorted(
                    comparisons,
                    key=lambda row: (row["metrics"]["profiled_whitened_rmse"], row["candidate_formula_id"]),
                )
                minimum = ordered[0]["metrics"]["profiled_whitened_rmse"]
                winners = sorted(
                    row["candidate_formula_id"]
                    for row in ordered
                    if math.isclose(
                        row["metrics"]["profiled_whitened_rmse"],
                        minimum,
                        abs_tol=float(config["scoring"]["winner_absolute_tolerance"]),
                        rel_tol=0.0,
                    )
                )
                second = ordered[1]["metrics"]["profiled_whitened_rmse"]
                gap = float(second - minimum)
                distinct = len(winners) == 1 and gap >= float(
                    config["scoring"]["minimum_whitened_gap_for_distinct_signature"]
                )
                recovered = truth_formula in winners
                recovered_count += int(recovered)
                distinct_count += int(recovered and distinct)
                recovery_by_truth[truth_formula]["scenarios"] += 1
                recovery_by_truth[truth_formula]["recovered"] += int(recovered)
                recovery_by_truth[truth_formula]["distinct"] += int(recovered and distinct)
                for winner in winners:
                    confusion_counts[truth_formula][winner] += 1
                completed_rows = []
                for binding in bindings:
                    decision = decide_scenario_eligibility(binding, catalogue, scenario)
                    ledger = ledger.append(
                        release=release,
                        binding=binding,
                        eligibility=decision,
                        adapter_sha256=(registration_by_formula[binding.formula_id].adapter_sha256 if binding.formula_id in registration_by_formula else None),
                        domain="strong-lens",
                        experiment_id=config["experiment_id"],
                    )
                    if decision.status is not EligibilityStatus.ELIGIBLE:
                        blocked_count += 1
                        continue
                    comparison = next(
                        row for row in comparisons if row["candidate_formula_id"] == binding.formula_id
                    )
                    status = status_from_result(
                        distinct_from_comparators=distinct,
                        self_injection_recovered=binding.formula_id == truth_formula and recovered,
                        numerical_valid=True,
                        powered=gap >= float(config["scoring"]["minimum_whitened_gap_for_distinct_signature"]),
                    )
                    diagnostics_payload = {
                        "candidate_formula_id": binding.formula_id,
                        "truth_formula_id": truth_formula,
                        "noise_family": family,
                        "winner_formula_ids": winners,
                        "profiled_whitened_gap": gap,
                        "real_response_used": False,
                        "lens_response_table_opened": False,
                    }
                    ledger = ledger.complete_last_eligible(
                        release=release,
                        binding=binding,
                        adapter_sha256=registration_by_formula[binding.formula_id].adapter_sha256,
                        domain="strong-lens",
                        experiment_id=config["experiment_id"],
                        status=status,
                        scenario_id=scenario.scenario_id,
                        object_id=scenario.object_id,
                        truth_world_id=truth_world_id,
                        seed_lineage_sha256=canonical_sha256(scenario.seed_lineage.to_dict()),
                        nuisance_draw=nuisance_draw,
                        parameter_cell_id="profiled-lens-scale-and-endpoint-clock-offset",
                        observable_ids=tuple(_RESPONSE_FOR[key] for key in _OUTPUTS),
                        result_sha256=comparison["output_sha256"],
                        metrics_sha256=_json_sha256(comparison["metrics"]),
                        diagnostics_sha256=_json_sha256(diagnostics_payload),
                        reason_codes=("response-blind", "source-cached-common-abi-execution", "synthetic-only"),
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
                        "lens": lens,
                        "truth_formula_id": truth_formula,
                        "truth_world_id": truth_world_id,
                        "noise": noise,
                        "value_locators": {
                            "responses": response_locators,
                            "variances": variance_locators,
                            "truth": {"key": truth_key, "sha256": array_sha256(truth_value)},
                            "path": VALUES_PATH.as_posix(),
                        },
                        "candidate_comparisons": completed_rows,
                        "injection_recovery": {
                            "primary_metric": config["scoring"]["primary_metric"],
                            "winner_formula_ids": winners,
                            "minimum_profiled_whitened_rmse": minimum,
                            "second_best_profiled_whitened_rmse": second,
                            "profiled_whitened_gap": gap,
                            "distinct_by_frozen_threshold": distinct,
                            "truth_recovered": recovered,
                            "truth_distinctly_recovered": recovered and distinct,
                        },
                    }
                )
    values_bytes = _npz_bytes(arrays)
    _require(values_bytes == _npz_bytes(arrays), "NPZ serialization nondeterministic")
    scenarios_bytes = b"".join(_json_bytes(row) + b"\n" for row in scenario_rows)
    ledger_bytes = _json_bytes(ledger.to_dict(), indent=2)
    confusion = {
        "schema": "open-gravity-lens-ray-clock-confusion-matrix-1.0",
        "truth_formula_ids": config["mechanisms"],
        "candidate_formula_ids": [row.formula_id for row in executable],
        "winner_membership_counts": confusion_counts,
        "recovery_by_truth": recovery_by_truth,
        "scenario_count": len(scenario_rows),
        "candidate_comparison_count": comparison_count,
        "truth_recovered_count": recovered_count,
        "distinct_truth_recovered_count": distinct_count,
        "numerical_failure_count": 0,
        "no_hand_ranking": True,
    }
    confusion_bytes = _json_bytes(confusion, indent=2)
    receipt_body = {
        "schema": "open-gravity-lens-ray-clock-source-shaped-synthetic-injection-matrix-receipt-2.0",
        "package_id": config["package_id"],
        "version": config["version"],
        "status": "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT",
        "claim_class": config["claim_class"],
        "scientific_claim": "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION",
        "independent_audit_completed": False,
        "distinct_independent_audit_required": True,
        "lens_count": len(source_items),
        "mechanism_count": len(executable),
        "noise_family_count": len(config["noise_families"]),
        "scenario_count": len(scenario_rows),
        "common_abi_execution_count": len(cache),
        "candidate_comparison_count": comparison_count,
        "replay_entry_count": len(ledger.entries),
        "blocked_ledger_entry_count": blocked_count,
        "truth_recovered_count": recovered_count,
        "distinct_truth_recovered_count": distinct_count,
        "recovery_by_truth": recovery_by_truth,
        "mechanism_ids": [row.formula_id for row in executable],
        "formula_bindings": {row.formula_id: row.to_dict() for row in bindings},
        "formula_binding_sha256": {row.formula_id: row.content_sha256 for row in bindings},
        "adapter_sha256": {row.formula_binding.formula_id: row.adapter_sha256 for row in registrations},
        "adapter_blocks": config["adapter_blocks"],
        "predecessor_binding": config["predecessor_binding"],
        "source_lenses": [{"lens": item["lens"], **item["metadata"]} for item in source_items],
        "source_anchor_sha256": anchor_hashes,
        "invariance_gates": {key: diagnostics[key] for key in diagnostics if key.startswith("maximum_") or key == "pass"},
        "identifiability": {
            "pair_count": diagnostics["pair_count"],
            "degenerate_pair_count": diagnostics["degenerate_pair_count"],
            "threshold": config["scoring"]["pairwise_degeneracy_profiled_rmse_max"],
        },
        "package_hashes": {
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
            "module_raw_sha256": module_sha,
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH),
        },
        "artifact_sha256": {
            "values.npz": hashlib.sha256(values_bytes).hexdigest(),
            "scenarios.jsonl": hashlib.sha256(scenarios_bytes).hexdigest(),
            "ledger.json": hashlib.sha256(ledger_bytes).hexdigest(),
            "confusion-matrix.json": hashlib.sha256(confusion_bytes).hexdigest(),
            "invariance-and-identifiability.json": hashlib.sha256(diagnostics_bytes).hexdigest(),
        },
        "access_accounting": config["access_contract"],
        "limitations": [
            "All responses are synthetic; no empirical support or rejection is authorized.",
            "The lens mass is the response-blind Lane1 exact point-mass lift with a deterministic stars+NFW source split, not a fitted extended lens model.",
            "Published image delays are source metadata only and are never used as a scoring response.",
            "Stationary GR predicts zero residual differential path redshift even though lensing, Shapiro delay, and Weyl exposure are nonzero.",
            "The endpoint and path laws, Phi/Psi slip, and geometric-delay control are synthetic directional signatures, not calibrated theories.",
            "ESO pixels/spectra, SLACS confirmation, raw lens response tables, moving-lens velocities, and microlensing maps remain sealed or absent.",
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
        if not (_ROOT / path).is_file() or (_ROOT / path).read_bytes() != payload:
            raise SystemExit(f"stored lens synthetic artifact differs: {path}")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze())
    else:
        receipt = check()
        print(receipt["status"], receipt["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
