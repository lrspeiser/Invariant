"""Item 57 unchanged-candidate tests on independent galaxy pipelines."""

from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import iv, kv

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_g0_experiment import _empirical_rar, _newtonian
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item56_disk_galaxy_gate import candidate_velocity

CONFIG_PATH = Path("configs/gravity_item57_independent_galaxy_gate_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM56_PATH = Path("runs/gravity/roadmap/item-56-disk-galaxy-gate-v1.json")
ITEM45_PATH = Path("runs/gravity/roadmap/item-45-universal-interactions-v1.json")
ITEM5_SOURCE_PATH = Path(
    "runs/gravity/roadmap/item-05-pressure-support-v1-source/"
    "exploration-source-manifest.json"
)
ITEM5_RAW_DIR = Path("work/item5-pressure-support-v1-raw")
THINGS_ARCHIVE_PATH = Path("work/item57-source-audit/things-0810.2100v2.tar")


class GravityItem57Error(RuntimeError):
    """Raised when the Item 57 freeze, data boundary, or replay changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item57-independent-galaxy-gate-config-1.0"
        or int(config.get("item", -1)) != 57
    ):
        raise GravityItem57Error("unexpected Item 57 config")
    if _sha256_file(root / GOAL_PATH) != config["stable_goal_sha256"]:
        raise GravityItem57Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem57Error("Item 57 scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem57Error("malformed Item 57 freeze binding")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != expected:
            raise GravityItem57Error(f"scientific dependency changed: {relative}")
    item56 = _read_json(root / ITEM56_PATH)
    predecessor = config["required_predecessor"]
    if item56["decision"] != predecessor["decision"]:
        raise GravityItem57Error("Item 56 decision changed")
    if bool(item56["counterexample_policy_assessment"]["terminal_rejection_in_tested_scope"]):
        raise GravityItem57Error("Item 56 unexpectedly terminalized the candidate")
    item45 = _read_json(root / ITEM45_PATH)["selected_candidate"]
    target = config["target_candidate"]
    if (
        int(item45["candidate_id"]) != int(target["candidate_id"])
        or int(item45["recipe_id"]) != int(target["recipe_id"])
        or item45["parameters"] != target["parameters"]
        or item45["interaction_expression"] != target["interaction_expression"]
        or target["refitting_allowed"]
        or int(target["post_freeze_formula_variants_allowed"]) != 0
    ):
        raise GravityItem57Error("unchanged Item 45 candidate binding changed")
    little = config["little_things"]
    exploration = {str(row["slug"]) for row in little["exploration_objects"]}
    reserved = set(map(str, little["reserved_confirmation_objects_never_read"]))
    if len(exploration) != 11 or len(reserved) != 5 or exploration & reserved:
        raise GravityItem57Error("LITTLE THINGS sample boundary changed")
    if (
        int(little["new_target_queries_allowed"]) != 0
        or int(little["reserved_predictor_queries_allowed"]) != 0
        or int(little["reserved_target_queries_allowed"]) != 0
        or int(little["target_rows_allowed"]) != 255
    ):
        raise GravityItem57Error("LITTLE THINGS sealed boundary changed")
    predictor = config["predictor_contract"]
    forbidden = (
        "galaxy_identifier_allowed_as_predictor",
        "observed_velocity_allowed_as_predictor",
        "uncertainty_allowed_as_predictor",
        "dark_matter_residual_allowed_as_predictor",
        "fitted_halo_allowed_as_predictor",
        "per_galaxy_formula_parameters_allowed",
    )
    if any(bool(predictor[name]) for name in forbidden):
        raise GravityItem57Error("Item 57 permits response leakage or retuning")
    counterexample = config["counterexample_policy"]
    if (
        counterexample["single_counterexample_terminal"]
        or counterexample["counterexample_count_alone_terminal"]
        or counterexample["finite_sample_may_prune_formula_family"]
        or not counterexample["failed_quality_gate_retains_exact_formula"]
    ):
        raise GravityItem57Error("Item 57 permits empirical over-pruning")
    confirmation = config["confirmation_boundary"]
    if (
        int(confirmation["sparc_confirmation_response_rows_allowed"]) != 0
        or confirmation["authorization_present"]
        or confirmation["confirmation_opened"]
    ):
        raise GravityItem57Error("SPARC confirmation boundary changed")
    things = config["things"]
    secondary = things["secondary_machine_readable_candidate"]
    if secondary["whole_archive_download_allowed"]:
        raise GravityItem57Error("secondary archive could expose sealed responses")
    if len(things["sample"]) != int(things["expected_rotation_curve_count"]):
        raise GravityItem57Error("THINGS sample count changed")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def build_preflight_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    source = _read_json(root / ITEM5_SOURCE_PATH)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item57-preflight-1.0",
            "item": 57,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "target_candidate": config["target_candidate"],
            "little_things_exploration_objects": config["little_things"][
                "exploration_objects"
            ],
            "existing_little_things_target_rows": sum(
                int(record["target"]["rows"]) for record in source["records"]
            ),
            "new_target_queries_allowed": 0,
            "reserved_confirmation_accesses_allowed": 0,
            "sparc_confirmation_response_rows_allowed": 0,
            "things_whole_secondary_archive_download_allowed": False,
            "post_freeze_formula_variants": 0,
            "paid_model_calls": 0,
        }
    )


def write_preflight_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root))
    return path


def _download(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise GravityItem57Error(f"predictor query failed: {url}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _normal_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def parse_photometry_payload(payload: bytes, *, expected_name: str) -> dict[str, float | str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem57Error("photometry response is not UTF-8") from exc
    header = "Name\tDist\tVMag\tRd\te_Rd"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem57Error(f"photometry schema changed: {expected_name}") from exc
    parsed: list[dict[str, float | str]] = []
    for line in lines[start + 1 :]:
        fields = line.split("\t")
        if len(fields) != 5:
            continue
        if not all(field.strip() for field in fields):
            continue
        try:
            parsed.append(
                {
                    "name": fields[0].strip(),
                    "distance_mpc": float(fields[1]),
                    "absolute_v_magnitude": float(fields[2]),
                    "disk_scale_kpc": float(fields[3]),
                    "disk_scale_error_kpc": float(fields[4]),
                }
            )
        except ValueError:
            continue
    matching = [
        row for row in parsed if _normal_name(str(row["name"])) == _normal_name(expected_name)
    ]
    if len(matching) != 1:
        raise GravityItem57Error(f"photometry query did not return one object: {expected_name}")
    row = matching[0]
    if (
        float(row["distance_mpc"]) <= 0.0
        or float(row["disk_scale_kpc"]) <= 0.0
        or float(row["disk_scale_error_kpc"]) < 0.0
    ):
        raise GravityItem57Error(f"invalid photometry: {expected_name}")
    return row


def acquire_photometry(root: Path) -> Path:
    config = load_config(root)
    records = []
    template = str(config["little_things"]["photometry_query_template"])
    source_dir = root / str(config["paths"]["source_dir"])
    for object_row in config["little_things"]["exploration_objects"]:
        slug = str(object_row["slug"])
        name = str(object_row["vizier_name"])
        url = template.format(name=urllib.parse.quote(name))
        raw_path = source_dir / "raw" / f"photometry-{slug}.tsv"
        payload = _download(url, raw_path)
        parsed = parse_photometry_payload(payload, expected_name=name)
        records.append(
            {
                "slug": slug,
                "query_name": name,
                "url": url,
                "path": str(raw_path.relative_to(root)).replace("\\", "/"),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "parsed": parsed,
            }
        )
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item57-photometry-source-1.0",
            "item": 57,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "catalog": "J/AJ/144/134/table1",
            "allowed_columns": config["little_things"]["photometry_allowed_columns"],
            "records": records,
            "counts": {
                "exploration_predictor_queries": len(records),
                "new_target_queries": 0,
                "reserved_predictor_queries": 0,
                "reserved_target_queries": 0,
                "sparc_confirmation_response_rows": 0,
            },
        }
    )
    path = _source_path(root, config, "photometry_manifest")
    _write_json(path, manifest)
    return path


def build_things_source_audit(root: Path) -> dict[str, Any]:
    config = load_config(root)
    archive = root / THINGS_ARCHIVE_PATH
    if _sha256_file(archive) != config["things"]["source_archive_sha256"]:
        raise GravityItem57Error("THINGS source archive changed")
    with tarfile.open(archive) as handle:
        members = {member.name: member.size for member in handle.getmembers() if member.isfile()}
    curve_figures = [f"figuur{number}.ps" for number in range(68, 87)]
    missing = [name for name in ["deblok_astroph.tex", *curve_figures] if name not in members]
    if missing:
        raise GravityItem57Error(f"THINGS source members missing: {missing}")
    things = config["things"]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item57-things-source-audit-1.0",
            "item": 57,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "primary_source": {
                "arxiv_identifier": things["arxiv_identifier"],
                "archive_path": str(THINGS_ARCHIVE_PATH),
                "archive_sha256": _sha256_file(archive),
                "rotation_curve_postscript_figures": curve_figures,
                "rotation_curve_figure_count": len(curve_figures),
                "machine_readable_radial_table_found": False,
            },
            "secondary_source_candidate": things[
                "secondary_machine_readable_candidate"
            ],
            "numeric_test_performed": False,
            "numeric_test_decision": things["numeric_test_if_source_not_machine_readable"],
            "reason_numeric_test_withheld": (
                "The primary release supplies plotted curves but no radial table. The only located "
                "machine-readable transcription is a whole archive that also contains sealed SPARC "
                "confirmation responses, and no selective endpoint was verified before freeze."
            ),
            "sample": things["sample"],
            "sparc_exploration_predictor_overlap": things[
                "sparc_exploration_predictor_overlap"
            ],
            "excluded_sparc_confirmation_overlap": things[
                "excluded_sparc_confirmation_overlap"
            ],
            "without_sparc_baryonic_predictor": things[
                "without_sparc_baryonic_predictor"
            ],
            "counts": {
                "primary_curve_figures": len(curve_figures),
                "numeric_things_galaxies": 0,
                "secondary_archive_bytes_downloaded": 0,
                "sealed_sparc_confirmation_rows_read": 0,
            },
            "claims": {
                "things_numeric_gate_passed": False,
                "source_format_limitation_retained": True,
                "same_object_claimed_independent_measurement_without_provenance": False,
            },
        }
    )


def write_things_source_audit(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "things_source_audit")
    _write_json(path, build_things_source_audit(root))
    return path


def _parse_predictor_surface_density(path: Path) -> tuple[np.ndarray, np.ndarray]:
    radii = []
    densities = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 12:
            raise GravityItem57Error(f"predictor schema changed: {path}")
        # Deliberately parse only R[kpc] and Sdens. Velocity columns remain strings.
        radius = float(fields[1])
        density = float(fields[10])
        if radius <= 0.0 or density < 0.0:
            raise GravityItem57Error(f"invalid surface-density predictor: {path}")
        radii.append(radius)
        densities.append(density)
    radius_array = np.asarray(radii, dtype=float)
    density_array = np.asarray(densities, dtype=float)
    if len(radius_array) < 2 or np.any(np.diff(radius_array) <= 0.0):
        raise GravityItem57Error(f"insufficient predictor radii: {path}")
    return radius_array, density_array


def _parse_existing_target(path: Path, *, expected_name: str) -> dict[str, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = "Name\tType\tR0.3\tV0.3\tR\tV\te_V"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem57Error(f"target schema changed: {expected_name}") from exc
    rows = []
    for line in lines[start + 1 :]:
        fields = line.split("\t")
        if len(fields) != 7 or fields[1].strip() != "Data":
            continue
        if _normal_name(fields[0]) != _normal_name(expected_name):
            raise GravityItem57Error("target query returned another galaxy")
        try:
            r03, v03, scaled_r, scaled_v, scaled_e = map(float, fields[2:])
        except ValueError:
            continue
        rows.append((r03 * scaled_r, v03 * scaled_v, v03 * scaled_e))
    rows.sort()
    if len(rows) < 2:
        raise GravityItem57Error(f"target rows missing: {expected_name}")
    result = np.asarray(rows, dtype=float)
    return {"radius": result[:, 0], "observed": result[:, 1], "sigma": result[:, 2]}


def gas_disk_velocity_squared(
    evaluation_radius: np.ndarray,
    density_radius: np.ndarray,
    surface_density: np.ndarray,
    *,
    neutral_gas_factor: float,
    softening_kpc: float,
    radial_subcells: int,
    azimuthal_cells: int,
    gravitational_constant: float,
) -> np.ndarray:
    """Integrate an axisymmetric softened annular disk without velocity inputs."""

    if (
        neutral_gas_factor <= 0.0
        or softening_kpc <= 0.0
        or radial_subcells < 2
        or azimuthal_cells < 32
    ):
        raise GravityItem57Error("invalid gas quadrature contract")
    edges = np.empty(len(density_radius) + 1, dtype=float)
    edges[1:-1] = 0.5 * (density_radius[:-1] + density_radius[1:])
    edges[0] = max(0.0, density_radius[0] - 0.5 * np.diff(density_radius[:2])[0])
    edges[-1] = density_radius[-1] + 0.5 * np.diff(density_radius[-2:])[0]
    annular_radii = []
    annular_widths = []
    annular_density = []
    for index, density in enumerate(surface_density):
        width = (edges[index + 1] - edges[index]) / radial_subcells
        annular_radii.extend(
            edges[index] + (np.arange(radial_subcells, dtype=float) + 0.5) * width
        )
        annular_widths.extend([width] * radial_subcells)
        annular_density.extend([density] * radial_subcells)
    source_r = np.asarray(annular_radii, dtype=float)
    source_width = np.asarray(annular_widths, dtype=float)
    sigma = np.asarray(annular_density, dtype=float) * neutral_gas_factor * 1.0e6
    phi = (np.arange(azimuthal_cells, dtype=float) + 0.5) * (2.0 * np.pi / azimuthal_cells)
    cosine = np.cos(phi)
    dphi = 2.0 * np.pi / azimuthal_cells
    result = np.empty_like(evaluation_radius, dtype=float)
    cell_mass_without_phi = sigma * source_r * source_width * dphi
    for index, radius in enumerate(evaluation_radius):
        separation2 = (
            radius * radius
            + source_r[:, None] ** 2
            - 2.0 * radius * source_r[:, None] * cosine[None, :]
            + softening_kpc * softening_kpc
        )
        inward = gravitational_constant * np.sum(
            cell_mass_without_phi[:, None]
            * (radius - source_r[:, None] * cosine[None, :])
            / np.power(separation2, 1.5)
        )
        result[index] = radius * inward
    if np.any(~np.isfinite(result)):
        raise GravityItem57Error("gas quadrature produced nonfinite values")
    return np.maximum(result, 0.0)


def exponential_disk_velocity_squared(
    radius: np.ndarray,
    *,
    disk_mass_msun: float,
    disk_scale_kpc: float,
    gravitational_constant: float,
) -> np.ndarray:
    if disk_mass_msun <= 0.0 or disk_scale_kpc <= 0.0:
        raise GravityItem57Error("invalid exponential stellar disk")
    y = radius / (2.0 * disk_scale_kpc)
    bracket = iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)
    result = (
        2.0
        * gravitational_constant
        * disk_mass_msun
        / disk_scale_kpc
        * y**2
        * bracket
    )
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise GravityItem57Error("stellar disk produced invalid values")
    return result


def _variant_parameters(
    variant: str, photometry: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    predictor = config["predictor_contract"]
    gas = float(predictor["nominal_neutral_gas_factor"])
    ml = float(predictor["nominal_vband_mass_to_light"])
    rd = float(photometry["disk_scale_kpc"])
    soft = float(predictor["nominal_gas_softening_fraction_of_rd"])
    systematic = config["evaluation"]["systematic_variants"]
    if variant == "nominal":
        pass
    elif variant.startswith("gas_factor_"):
        gas = float(variant.removeprefix("gas_factor_"))
    elif variant.startswith("vband_ml_"):
        ml = float(variant.removeprefix("vband_ml_"))
    elif variant == "rd_minus":
        rd -= float(photometry["disk_scale_error_kpc"])
    elif variant == "rd_plus":
        rd += float(photometry["disk_scale_error_kpc"])
    elif variant.startswith("gas_softening_"):
        soft = float(variant.removeprefix("gas_softening_"))
    else:
        raise GravityItem57Error(f"unknown systematic variant: {variant}")
    if rd <= 0.0:
        raise GravityItem57Error("disk-scale uncertainty endpoint is non-positive")
    allowed = {
        *(f"gas_factor_{float(value):g}" for value in systematic["neutral_gas_factors"]),
        *(f"vband_ml_{float(value):g}" for value in systematic["vband_mass_to_light"]),
        "rd_minus",
        "rd_plus",
        *(
            f"gas_softening_{float(value):g}"
            for value in systematic["gas_softening_fractions_of_rd"]
        ),
    }
    if variant != "nominal" and variant not in allowed:
        raise GravityItem57Error("systematic variant is outside the freeze")
    return gas, ml, rd, soft


def _variant_ids(config: Mapping[str, Any]) -> list[str]:
    systematic = config["evaluation"]["systematic_variants"]
    return [
        *(f"gas_factor_{float(value):g}" for value in systematic["neutral_gas_factors"]),
        *(f"vband_ml_{float(value):g}" for value in systematic["vband_mass_to_light"]),
        "rd_minus",
        "rd_plus",
        *(
            f"gas_softening_{float(value):g}"
            for value in systematic["gas_softening_fractions_of_rd"]
        ),
    ]


def _loss(prediction: np.ndarray, observed: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(np.square((prediction - observed) / sigma)))


def _predictions(
    radius: np.ndarray,
    density_radius: np.ndarray,
    surface_density: np.ndarray,
    photometry: Mapping[str, Any],
    config: Mapping[str, Any],
    variant: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    gas_factor, ml, rd, soft_fraction = _variant_parameters(variant, photometry, config)
    predictor = config["predictor_contract"]
    gravitational_constant = float(predictor["gravitational_constant_kpc_km2_s2_msun"])
    gas_v2 = gas_disk_velocity_squared(
        radius,
        density_radius,
        surface_density,
        neutral_gas_factor=gas_factor,
        softening_kpc=soft_fraction * rd,
        radial_subcells=int(predictor["gas_radial_subcells"]),
        azimuthal_cells=int(predictor["gas_azimuthal_cells"]),
        gravitational_constant=gravitational_constant,
    )
    luminosity = 10.0 ** (
        -0.4
        * (
            float(photometry["absolute_v_magnitude"])
            - float(predictor["solar_vband_absolute_magnitude"])
        )
    )
    stellar_v2 = exponential_disk_velocity_squared(
        radius,
        disk_mass_msun=ml * luminosity,
        disk_scale_kpc=rd,
        gravitational_constant=gravitational_constant,
    )
    vbar2 = np.maximum(gas_v2 + stellar_v2, 1.0e-12)
    effective_radius = 1.67834699 * rd
    a0 = float(predictor["acceleration_scale_km2_s2_kpc"])
    return (
        {
            "item45_geometry_density": candidate_velocity(
                radius, vbar2, effective_radius, config["target_candidate"], a0
            ),
            "newtonian_baryons": _newtonian(vbar2),
            "empirical_rar": _empirical_rar(radius, vbar2, a0),
        },
        {
            "neutral_gas_factor": gas_factor,
            "vband_mass_to_light": ml,
            "disk_scale_kpc": rd,
            "effective_radius_kpc": effective_radius,
            "gas_softening_kpc": soft_fraction * rd,
        },
    )


def _sign_changes_after_leave_one(values: np.ndarray) -> bool:
    if len(values) <= 1:
        return True
    full = float(np.mean(values))
    return any(full * float(np.mean(np.delete(values, index))) <= 0.0 for index in range(len(values)))


def _trim_changes_sign(values: np.ndarray, fraction: float) -> tuple[bool, float, int]:
    count = math.floor(len(values) * fraction)
    if count == 0 or 2 * count >= len(values):
        return False, float(np.mean(values)), 0
    ordered = np.sort(values)
    trimmed = ordered[count:-count]
    full = float(np.mean(values))
    mean = float(np.mean(trimmed))
    return full * mean <= 0.0, mean, count


def build_little_things_evaluation(root: Path) -> dict[str, Any]:
    config = load_config(root)
    photometry_manifest = _read_json(_source_path(root, config, "photometry_manifest"))
    photometry_by_slug = {
        str(record["slug"]): record["parsed"] for record in photometry_manifest["records"]
    }
    source = _read_json(root / ITEM5_SOURCE_PATH)
    source_by_slug = {str(record["galaxy"]): record for record in source["records"]}
    variants = _variant_ids(config)
    model_ids = ("item45_geometry_density", "newtonian_baryons", "empirical_rar")
    nominal_losses: dict[str, dict[str, float]] = {name: {} for name in model_ids}
    variant_losses: dict[str, dict[str, dict[str, float]]] = {
        variant: {name: {} for name in model_ids} for variant in variants
    }
    per_galaxy = []
    missing = []
    total_existing_target_rows = 0
    total_evaluated_rows = 0
    quadrature_point_cells = 0
    for object_row in config["little_things"]["exploration_objects"]:
        slug = str(object_row["slug"])
        name = str(object_row["vizier_name"])
        record = source_by_slug[slug]
        predictor_path = root / str(record["predictor"]["path"])
        target_path = root / str(record["target"]["path"])
        if _sha256_file(predictor_path) != record["predictor"]["sha256"]:
            raise GravityItem57Error(f"predictor file changed: {slug}")
        if _sha256_file(target_path) != record["target"]["sha256"]:
            raise GravityItem57Error(f"target file changed: {slug}")
        density_radius, surface_density = _parse_predictor_surface_density(predictor_path)
        target = _parse_existing_target(target_path, expected_name=name)
        total_existing_target_rows += len(target["radius"])
        valid = (
            (target["radius"] > 0.0)
            & (target["radius"] <= density_radius[-1])
            & (target["observed"] >= 0.0)
            & (target["sigma"] > 0.0)
            & np.isfinite(target["radius"])
            & np.isfinite(target["observed"])
            & np.isfinite(target["sigma"])
        )
        radius = target["radius"][valid]
        observed = target["observed"][valid]
        sigma = target["sigma"][valid]
        minimum = int(config["evaluation"]["minimum_target_rows_per_galaxy"])
        if len(radius) < minimum:
            missing.append(
                {
                    "galaxy": slug,
                    "reason": "insufficient_target_rows_inside_surface_density_support",
                    "existing_target_rows": len(target["radius"]),
                    "evaluated_rows": len(radius),
                }
            )
            continue
        total_evaluated_rows += len(radius)
        nominal, parameters = _predictions(
            radius,
            density_radius,
            surface_density,
            photometry_by_slug[slug],
            config,
            "nominal",
        )
        scores = {model: _loss(prediction, observed, sigma) for model, prediction in nominal.items()}
        for model in model_ids:
            nominal_losses[model][slug] = scores[model]
        systematics = {}
        for variant in variants:
            varied, varied_parameters = _predictions(
                radius,
                density_radius,
                surface_density,
                photometry_by_slug[slug],
                config,
                variant,
            )
            systematics[variant] = {
                "parameters": varied_parameters,
                "losses": {
                    model: _loss(prediction, observed, sigma)
                    for model, prediction in varied.items()
                },
            }
            for model in model_ids:
                variant_losses[variant][model][slug] = systematics[variant]["losses"][model]
        quadrature_point_cells += (
            (1 + len(variants))
            * len(radius)
            * len(density_radius)
            * int(config["predictor_contract"]["gas_radial_subcells"])
            * int(config["predictor_contract"]["gas_azimuthal_cells"])
        )
        per_galaxy.append(
            {
                "galaxy": slug,
                "published_name": name,
                "source_density_rows": len(density_radius),
                "existing_target_rows": len(target["radius"]),
                "evaluated_rows": len(radius),
                "response_blind_parameters": parameters,
                "losses": scores,
                "candidate_beats_newton": scores["item45_geometry_density"]
                < scores["newtonian_baryons"],
                "candidate_beats_rar": scores["item45_geometry_density"]
                < scores["empirical_rar"],
                "systematics": systematics,
            }
        )
    names = sorted(nominal_losses["item45_geometry_density"])
    aggregate = {
        model: {
            "equal_galaxy_mean_squared_standardized_residual": float(
                np.mean([nominal_losses[model][name] for name in names])
            ),
            "median_galaxy_mean_squared_standardized_residual": float(
                np.median([nominal_losses[model][name] for name in names])
            ),
        }
        for model in model_ids
    }
    variant_aggregate = {
        variant: {
            "scores": {
                model: {
                    "equal_galaxy_mean_squared_standardized_residual": float(
                        np.mean([variant_losses[variant][model][name] for name in names])
                    )
                }
                for model in model_ids
            },
            "candidate_loses_to_rar": bool(
                np.mean([variant_losses[variant]["item45_geometry_density"][name] for name in names])
                > np.mean([variant_losses[variant]["empirical_rar"][name] for name in names])
            ),
            "candidate_galaxy_wins_vs_rar": sum(
                variant_losses[variant]["item45_geometry_density"][name]
                < variant_losses[variant]["empirical_rar"][name]
                for name in names
            ),
        }
        for variant in variants
    }
    difference_vs_rar = np.asarray(
        [
            nominal_losses["empirical_rar"][name]
            - nominal_losses["item45_geometry_density"][name]
            for name in names
        ]
    )
    leave_one_changes = _sign_changes_after_leave_one(difference_vs_rar)
    trim_changes, trimmed_mean, trimmed_each_tail = _trim_changes_sign(
        difference_vs_rar, float(config["evaluation"]["influence_trim_fraction"])
    )
    raw_counterexamples = [name for name, delta in zip(names, difference_vs_rar) if delta < 0.0]
    stable_counterexamples = [
        name
        for name in raw_counterexamples
        if all(
            variant_losses[variant]["item45_geometry_density"][name]
            > variant_losses[variant]["empirical_rar"][name]
            for variant in variants
        )
    ]
    quality_gate = (
        len(names) >= int(config["evaluation"]["minimum_quality_galaxies"])
        and total_evaluated_rows >= int(config["evaluation"]["minimum_quality_rows"])
        and total_existing_target_rows == int(config["little_things"]["target_rows_allowed"])
        and photometry_manifest["counts"]["new_target_queries"] == 0
        and photometry_manifest["counts"]["reserved_predictor_queries"] == 0
        and photometry_manifest["counts"]["reserved_target_queries"] == 0
    )
    threshold = config["evaluation"]["negative_replication_gates"]
    galaxy_loss_fraction = len(raw_counterexamples) / len(names)
    systematic_losses = sum(row["candidate_loses_to_rar"] for row in variant_aggregate.values())
    negative_gates = {
        "quality_gate_passed": quality_gate,
        "candidate_loses_equal_galaxy_to_newton": aggregate["item45_geometry_density"][
            "equal_galaxy_mean_squared_standardized_residual"
        ]
        > aggregate["newtonian_baryons"]["equal_galaxy_mean_squared_standardized_residual"],
        "candidate_loses_equal_galaxy_to_rar": aggregate["item45_geometry_density"][
            "equal_galaxy_mean_squared_standardized_residual"
        ]
        > aggregate["empirical_rar"]["equal_galaxy_mean_squared_standardized_residual"],
        "galaxy_fraction_losing_to_rar_minimum": galaxy_loss_fraction
        >= float(threshold["galaxy_fraction_losing_to_rar_minimum"]),
        "systematic_variants_losing_to_rar_minimum": systematic_losses
        >= int(threshold["systematic_variants_losing_to_rar_minimum"]),
        "leave_one_and_trim_preserve_negative_sign": not leave_one_changes
        and not trim_changes,
    }
    negative_replication = all(negative_gates.values())
    survival_threshold = config["evaluation"]["unexpected_survival_gate"]
    survival_gates = {
        "candidate_beats_equal_galaxy_rar": aggregate["item45_geometry_density"][
            "equal_galaxy_mean_squared_standardized_residual"
        ]
        < aggregate["empirical_rar"]["equal_galaxy_mean_squared_standardized_residual"],
        "galaxy_fraction_beating_rar_minimum": 1.0 - galaxy_loss_fraction
        >= float(survival_threshold["galaxy_fraction_beating_rar_minimum"]),
        "systematic_variants_beating_rar_minimum": len(variants) - systematic_losses
        >= int(survival_threshold["systematic_variants_beating_rar_minimum"]),
    }
    improvement_vs_newton = 100.0 * (
        aggregate["newtonian_baryons"]["equal_galaxy_mean_squared_standardized_residual"]
        - aggregate["item45_geometry_density"][
            "equal_galaxy_mean_squared_standardized_residual"
        ]
    ) / aggregate["newtonian_baryons"]["equal_galaxy_mean_squared_standardized_residual"]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(names),
        "raw_counterexample_count": len(raw_counterexamples),
        "quality_verified_counterexample_count": len(raw_counterexamples) if quality_gate else 0,
        "uncertainty_resolved_counterexample_count": len(stable_counterexamples)
        if quality_gate
        else 0,
        "aggregate_improvement_percent": improvement_vs_newton,
        "quality_gate_passed": quality_gate,
        "strongest_baseline_failed": negative_gates["candidate_loses_equal_galaxy_to_rar"],
        "leave_one_changes_sign": leave_one_changes,
        "trim_changes_sign": trim_changes,
        "independent_failure_strata": 1 if negative_replication else 0,
        "unchanged_independent_replication_failures": 1 if negative_replication else 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    assessment = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item57-little-things-evaluation-1.0",
            "item": 57,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "target_candidate": config["target_candidate"],
            "predictor_contract": config["predictor_contract"],
            "aggregate_scores": aggregate,
            "systematic_variants": variant_aggregate,
            "negative_replication_gates": negative_gates,
            "negative_replication_gate_passed": negative_replication,
            "unexpected_survival_gates": survival_gates,
            "unexpected_survival_gate_passed": all(survival_gates.values()),
            "candidate_improvement_over_newton_equal_galaxy_percent": improvement_vs_newton,
            "candidate_galaxy_wins_vs_rar": len(names) - len(raw_counterexamples),
            "candidate_galaxy_losses_vs_rar": len(raw_counterexamples),
            "counterexamples_stable_across_all_systematics": stable_counterexamples,
            "influence": {
                "leave_one_changes_sign": leave_one_changes,
                "symmetric_trim_changes_sign": trim_changes,
                "trimmed_mean_rar_minus_candidate_loss": trimmed_mean,
                "trimmed_each_tail": trimmed_each_tail,
            },
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": assessment,
            "per_galaxy": per_galaxy,
            "missing_or_quality_limited": missing,
            "counts": {
                "authorized_exploration_galaxies": 11,
                "evaluable_galaxies": len(names),
                "existing_target_rows": total_existing_target_rows,
                "evaluated_rows": total_evaluated_rows,
                "systematic_variants": len(variants),
                "post_freeze_formula_variants": 0,
                "new_target_queries": 0,
                "reserved_predictor_queries": 0,
                "reserved_target_queries": 0,
                "sparc_confirmation_response_rows": 0,
            },
            "compute": {
                "backend": "numpy_scipy_cpu",
                "gas_quadrature_point_cells": quadrature_point_cells,
                "gpu_used": False,
                "paid_model_calls": 0,
                "paid_api_cost_usd": 0.0,
            },
            "claims": {
                "little_things_unchanged_candidate_test_completed": True,
                "negative_replication_gate_passed": negative_replication,
                "unexpected_survival_gate_passed": all(survival_gates.values()),
                "fresh_sparc_confirmation_completed": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": [
                "The target and HI surface-density profiles come from separate reductions of the LITTLE THINGS survey; they are response-distinct but not instrument-independent.",
                "The baryonic reconstruction assumes axisymmetry, a softened thin gas disk, an exponential stellar disk, and frozen V-band mass-to-light brackets.",
                "Published target errors do not provide a complete distance, inclination, beam, and radial covariance matrix.",
                "Only radii inside the published HI surface-density support are scored; the rule was frozen before evaluation.",
                "A replicated failure can retire only the exact representation in this tested scope and never the broader geometry-density family.",
            ],
        }
    )


def write_little_things_evaluation(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "little_things_evaluation")
    _write_json(path, build_little_things_evaluation(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    photometry = _read_json(_source_path(root, config, "photometry_manifest"))
    things = _read_json(_source_path(root, config, "things_source_audit"))
    little = _read_json(_source_path(root, config, "little_things_evaluation"))
    negative = bool(little["negative_replication_gate_passed"])
    survival = bool(little["unexpected_survival_gate_passed"])
    if negative:
        decision = (
            "ITEM57_LITTLE_THINGS_REPLICATES_NEGATIVE_THINGS_SOURCE_LIMITED_"
            "EXACT_REPRESENTATION_RETIRED_IN_TESTED_SCOPE"
        )
    elif survival:
        decision = "ITEM57_LITTLE_THINGS_UNEXPECTED_SURVIVAL_THINGS_SOURCE_LIMITED_LEAD_RETAINED"
    elif not little["negative_replication_gates"]["quality_gate_passed"]:
        decision = "ITEM57_INDEPENDENT_GALAXY_GATE_QUALITY_LIMITED_RETAINED"
    else:
        decision = "ITEM57_MIXED_LITTLE_THINGS_RESULT_THINGS_SOURCE_LIMITED_RETAINED"
    source_bindings = {}
    for label, key in (
        ("preflight", "preflight_manifest"),
        ("photometry", "photometry_manifest"),
        ("things_audit", "things_source_audit"),
        ("little_things_evaluation", "little_things_evaluation"),
    ):
        path = _source_path(root, config, key)
        source_bindings[label] = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256_file(path),
        }
    source_bindings["config"] = {
        "path": str(CONFIG_PATH),
        "sha256": _sha256_file(root / CONFIG_PATH),
    }
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item57-independent-galaxy-result-1.0",
            "item": 57,
            "goal": "GRAVITY_ROADMAP_ITEM_57_INDEPENDENT_GALAXY_GATE",
            "decision": decision,
            "target_candidate": config["target_candidate"],
            "little_things": {
                "aggregate_scores": little["aggregate_scores"],
                "negative_replication_gates": little["negative_replication_gates"],
                "negative_replication_gate_passed": negative,
                "unexpected_survival_gate_passed": survival,
                "candidate_galaxy_wins_vs_rar": little["candidate_galaxy_wins_vs_rar"],
                "candidate_galaxy_losses_vs_rar": little["candidate_galaxy_losses_vs_rar"],
                "counterexample_policy_assessment": little[
                    "counterexample_policy_assessment"
                ],
                "counts": little["counts"],
            },
            "things": {
                "numeric_test_performed": things["numeric_test_performed"],
                "numeric_test_decision": things["numeric_test_decision"],
                "counts": things["counts"],
            },
            "full_two_pipeline_numeric_gate_passed": False,
            "source_bindings": source_bindings,
            "claims": {
                "roadmap_item_57_execution_complete": True,
                "full_two_pipeline_numeric_gate_passed": False,
                "little_things_unchanged_candidate_test_completed": True,
                "things_numeric_test_completed": False,
                "fresh_sparc_confirmation_completed": False,
                "exact_representation_retired_in_tested_scope": negative,
                "formula_family_pruned": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": [*little["limitations"], things["reason_numeric_test_withheld"]],
            "next_action": (
                "Preserve every object-level result and the broader geometry-density family. "
                "Advance to Item 58 while retaining a THINGS data-procurement follow-up; do not "
                "open the sealed SPARC confirmation set without authorization."
            ),
            "preflight": preflight,
            "photometry_counts": photometry["counts"],
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    checks = {
        "preflight": _read_json(_source_path(root, config, "preflight_manifest"))
        == build_preflight_manifest(root),
        "things_source_audit": _read_json(_source_path(root, config, "things_source_audit"))
        == build_things_source_audit(root),
        "little_things_evaluation": _read_json(
            _source_path(root, config, "little_things_evaluation")
        )
        == build_little_things_evaluation(root),
        "aggregate": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "acquire", "things-audit", "evaluate", "aggregate", "replay"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight_manifest(root))
    elif args.command == "acquire":
        result = str(acquire_photometry(root))
    elif args.command == "things-audit":
        result = str(write_things_source_audit(root))
    elif args.command == "evaluate":
        result = str(write_little_things_evaluation(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
