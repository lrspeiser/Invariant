"""Build five response-blind THINGS/HERACLES/S4G model-lifted source families."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.wcs import NoConvergence

from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as base,
)

CONFIG_PATH = Path(
    "configs/open_gravity_rg_things_heracles_s4g_model_lifted_3d_source_builder_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_heracles_s4g_model_lifted_3d_source_builder_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_rg_things_heracles_s4g_model_lifted_3d_source_builder_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-heracles-s4g-model-lifted-3d-source-builder-v1/receipt.json"
)

_CONFIG_RAW_SHA256 = "2cefafadf5eb5992ce75e3dae88dede88c7596d3272139585c716ba083432e0b"
_CONFIG_CONTENT_SHA256 = "147c22be007dedaae905a9b97522ffe804c370f1a650dc53ed3f130d40a65f36"
_MODULE_SEMANTIC_SHA256 = "b1c6319cc0791a677fb4271f2e2ceea5202b828d1b33a70ed021f522d22f5073"
_TEST_RAW_SHA256 = "b7b3fd6aed1675ea4dbb2b87feea17a78616770343f940c5b30e15e32a17b445"
_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f*]{64}("\r?\n)')
_SCHEMA = "invariant-open-gravity-rg-things-heracles-s4g-model-lifted-3d-source-builder-1.0"
_PROFILE_SCHEMA = (
    "invariant-open-gravity-rg-things-heracles-s4g-model-lifted-3d-source-profiles-1.0"
)
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-rg-things-heracles-s4g-model-lifted-3d-source-builder-receipt-1.0"
)
_OBJECTS = ("NGC2903", "NGC2976", "NGC3198", "NGC3521", "NGC4214")
_ROLES = (
    "STELLAR_MASS_MAP",
    "STELLAR_ICA_MASK",
    "STELLAR_COLOR_MAP",
    "HI_MOM0_NATURAL",
    "HI_MOM0_ROBUST",
    "CO21_MOM0",
    "CO21_EMOM0",
)


class SourceBuilderError(RuntimeError):
    """Raised when the response-blind builder contract changes or fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceBuilderError(message)


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
        raise SourceBuilderError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: dict[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "RESPONSE_BLIND_FIVE_OBJECT_MODEL_LIFTED_2P5D_SOURCE_BUILDER",
        "status changed",
    )
    _require(config["objects"] == list(_OBJECTS), "object ledger changed")
    inventory = config["source_inventory"]
    _require(inventory["exact_file_count"] == 35, "source count changed")
    _require(inventory["exact_source_bytes"] == 90_449_926, "source bytes changed")
    _require(inventory["required_roles"] == list(_ROLES), "source roles changed")
    cells = config["cell_contract"]
    _require(cells["primary_cartesian_cells_per_object"] == 72, "primary cells changed")
    _require(cells["reliable_object_cells"] == 77, "reliable cell count changed")
    _require(cells["uncertain_object_cells"] == 81, "uncertain cell count changed")
    _require(cells["total_cells"] == 393, "total cell count changed")
    _require(cells["response_based_cell_selection"] is False, "response selection enabled")
    _require(cells["retain_every_failure"] is True, "failure retention lost")
    anchors = config["published_anchor_contract"]
    _require(len(anchors["measurement_sources"]) == 4, "measurement source removed")
    _require(len(anchors["independent_benchmarks"]) == 3, "benchmark removed")
    _require(len(anchors["mandatory_before_response"]) == 8, "gate removed")
    boundary = config["scientific_boundary"]
    _require(boundary["source_files_opened"] == 35, "source accounting changed")
    _require(boundary["source_bytes_opened"] == 90_449_926, "byte accounting changed")
    _require(boundary["response_or_velocity_files_opened"] == 0, "response access enabled")
    _require(boundary["response_rows_opened"] == 0, "response rows enabled")
    _require(boundary["scores_computed"] == boundary["models_fit"] == 0, "scoring enabled")
    _require(boundary["observed_full_3d_geometry"] is False, "3D source overclaim")
    _require(boundary["model_lifted_2p5d_only"] is True, "model-lift label lost")
    claims = config["claims"]
    _require(claims["five_real_source_sets_bound"] is True, "source claim lost")
    _require(claims["model_lifted_source_profiles_derived"] is True, "profile claim lost")
    _require(claims["geometry_and_vertical_systematics_enumerated"] is True, "systematics lost")
    _require(
        not any(
            claims[key]
            for key in (
                "observed_full_3d_sources",
                "scientific_response_scored",
                "refracted_gravity_supported",
                "new_gravity_law_established",
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


def _load_binding(binding: dict[str, Any], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for role in ("config", "module", "test", "receipt"):
        path = _repo_path(binding[f"{role}_path"])
        _require(file_sha256(path) == binding[f"{role}_raw_sha256"], f"{label} {role} changed")
    config = _read_json(_repo_path(binding["config_path"]), f"{label} config")
    receipt = _read_json(_repo_path(binding["receipt_path"]), f"{label} receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"],
        f"{label} receipt content changed",
    )
    return config, receipt


def _load_dependencies(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _acquisition_config, acquisition_receipt = _load_binding(
        config["source_acquisition_binding"], "source acquisition"
    )
    geometry_config, geometry_receipt = _load_binding(config["geometry_binding"], "geometry")
    _operator_config, operator_receipt = _load_binding(
        config["operator_benchmark_binding"], "operator benchmark"
    )
    _require(
        acquisition_receipt["scientific_boundary"]["response_rows_opened"] == 0,
        "source receipt exposed response",
    )
    _require(
        geometry_receipt["access_state"]["response_rows_opened"] == 0,
        "geometry receipt exposed response",
    )
    _require(
        operator_receipt["access_state"]["response_rows_opened"] == 0,
        "operator receipt exposed response",
    )
    return acquisition_receipt, geometry_config, operator_receipt


def _source_paths(
    config: dict[str, Any], acquisition_receipt: dict[str, Any]
) -> dict[tuple[str, str], Path]:
    records = [
        row for row in acquisition_receipt["inventory"]["records"] if row["object_id"] in _OBJECTS
    ]
    _require(len(records) == config["source_inventory"]["exact_file_count"], "source rows changed")
    _require(
        sum(row["bytes"] for row in records) == config["source_inventory"]["exact_source_bytes"],
        "source row bytes changed",
    )
    counts: dict[str, int] = {}
    paths: dict[tuple[str, str], Path] = {}
    for row in records:
        counts[row["survey"]] = counts.get(row["survey"], 0) + 1
        key = (row["object_id"], row["role"])
        _require(key not in paths, "duplicate source role")
        path = _repo_path(row["relative_path"])
        _require(path.is_file(), "source file missing")
        _require(path.stat().st_size == row["bytes"], "source file bytes changed")
        _require(file_sha256(path) == row["sha256"], "source file hash changed")
        paths[key] = path
    _require(counts == config["source_inventory"]["survey_role_counts"], "survey counts changed")
    for object_id in _OBJECTS:
        _require(
            {role for (name, role) in paths if name == object_id} == set(_ROLES),
            "object source roles changed",
        )
    return paths


def _load_images(
    object_id: str, paths: dict[tuple[str, str], Path]
) -> dict[str, tuple[np.ndarray, Any]]:
    aliases = {
        "STELLAR_FLUX": "STELLAR_MASS_MAP",
        "STELLAR_ICA_MASK": "STELLAR_ICA_MASK",
        "STELLAR_COLOR": "STELLAR_COLOR_MAP",
        "HI_MOM0_NATURAL_SENSITIVITY": "HI_MOM0_NATURAL",
        "HI_MOM0_ROBUST_PRIMARY": "HI_MOM0_ROBUST",
        "CO21_BROAD_MOM0": "CO21_MOM0",
        "CO21_BROAD_EMOM0": "CO21_EMOM0",
    }
    return {
        target: base._fits_image(paths[(object_id, source)]) for target, source in aliases.items()
    }


def inclination_deg(ellipticity: float, intrinsic_q0: float) -> float:
    observed_q = 1.0 - ellipticity
    if observed_q <= intrinsic_q0:
        return 90.0
    cosine2 = (observed_q * observed_q - intrinsic_q0 * intrinsic_q0) / (
        1.0 - intrinsic_q0 * intrinsic_q0
    )
    return math.degrees(math.acos(math.sqrt(max(min(cosine2, 1.0), 0.0))))


def _metadata_variant(
    row: dict[str, Any],
    *,
    variant_id: str,
    q0: float,
    pa_offset_sigma: float = 0.0,
    ell_offset_sigma: float = 0.0,
) -> dict[str, Any]:
    ellipticity = float(row["outer_ellipticity"]) + ell_offset_sigma * float(
        row["outer_ellipticity_sd"]
    )
    _require(0.0 <= ellipticity < 1.0, "geometry ellipticity escaped physical range")
    return {
        "object_id": row["object_id"],
        "ra_deg": float(row["ra_deg"]),
        "dec_deg": float(row["dec_deg"]),
        "distance_mpc": float(row["distance_mpc"]),
        "position_angle_deg": float(row["outer_position_angle_deg"])
        + pa_offset_sigma * float(row["outer_position_angle_sd_deg"]),
        "inclination_deg": inclination_deg(ellipticity, q0),
        "ellipticity": ellipticity,
        "intrinsic_q0": q0,
        "geometry_variant_id": variant_id,
        "orientation_flag": row["orientation_flag"],
    }


def geometry_variants(config: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    central_q0 = float(config["cell_contract"]["central_geometry_intrinsic_q0"])
    variants = [
        _metadata_variant(row, variant_id="CENTRAL_Q0_0P13", q0=central_q0),
        _metadata_variant(row, variant_id="Q0_0P0", q0=0.0),
        _metadata_variant(row, variant_id="Q0_0P2", q0=0.2),
    ]
    if row["orientation_flag"] != "ok":
        variants.extend(
            [
                _metadata_variant(
                    row, variant_id="PA_MINUS_1SIGMA", q0=central_q0, pa_offset_sigma=-1.0
                ),
                _metadata_variant(
                    row, variant_id="PA_PLUS_1SIGMA", q0=central_q0, pa_offset_sigma=1.0
                ),
                _metadata_variant(
                    row,
                    variant_id="ELL_MINUS_1SIGMA",
                    q0=central_q0,
                    ell_offset_sigma=-1.0,
                ),
                _metadata_variant(
                    row,
                    variant_id="ELL_PLUS_1SIGMA",
                    q0=central_q0,
                    ell_offset_sigma=1.0,
                ),
            ]
        )
    return variants


def _maps(
    config: dict[str, Any],
    metadata: dict[str, Any],
    images: dict[str, tuple[np.ndarray, Any]],
    *,
    beam: str = "ROBUST_PRIMARY",
    n: int | None = None,
    box_kpc: float | None = None,
    use_sip: bool = False,
) -> dict[str, Any]:
    return base._surface_maps(
        config,
        metadata,
        images,
        n=int(config["map_transform"]["primary_grid_pixels"] if n is None else n),
        box_kpc=float(config["map_transform"]["primary_box_kpc"] if box_kpc is None else box_kpc),
        beam=beam,
        use_sip=use_sip,
    )


def _scale_length(maps: dict[str, Any]) -> tuple[float, float]:
    rhalf_pc = base._half_mass_radius_pc(
        maps["stellar_fixed"], maps["x_pc"], maps["y_pc"], float(maps["dx_pc"])
    )
    _require(300.0 < rhalf_pc < 20_000.0, "stellar half-mass radius outside source sanity")
    return rhalf_pc, rhalf_pc / 1.678


def _control_cell(
    config: dict[str, Any],
    metadata: dict[str, Any],
    maps: dict[str, Any],
    *,
    cell_id: str,
) -> tuple[dict[str, Any], list[dict[str, float]], float, float]:
    rhalf_pc, rd_pc = _scale_length(maps)
    middle_ratio = float(
        config["vertical_and_gravity_model"]["stellar_height_over_exponential_scale_cells"][1]
    )
    summary, profile = base._build_cell(
        config,
        metadata,
        maps,
        cell_id=cell_id,
        stellar_ml="FIXED_0P6",
        co_source="WITH_CO",
        hstar_pc=rd_pc * middle_ratio,
        hgas_pc=200.0,
        cache={},
    )
    return summary, profile, rhalf_pc, rd_pc


def _build_object(
    config: dict[str, Any],
    geometry_row: dict[str, Any],
    images: dict[str, tuple[np.ndarray, Any]],
) -> dict[str, Any]:
    variants = geometry_variants(config, geometry_row)
    central = variants[0]
    primary_maps = _maps(config, central, images)
    rhalf_pc, rd_pc = _scale_length(primary_maps)
    gravity = config["vertical_and_gravity_model"]
    cells: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []

    for beam in config["cell_contract"]["primary_cartesian_axes"]["beam"]:
        maps = (
            primary_maps if beam == "ROBUST_PRIMARY" else _maps(config, central, images, beam=beam)
        )
        cache: dict[tuple[str, float], np.ndarray] = {}
        for stellar_ml in config["cell_contract"]["primary_cartesian_axes"]["stellar_ml"]:
            for co_source in config["cell_contract"]["primary_cartesian_axes"]["co_source"]:
                for ratio in gravity["stellar_height_over_exponential_scale_cells"]:
                    for hgas_pc in gravity["gas_height_pc_cells"]:
                        cell_id = (
                            f"{beam}:{stellar_ml}:{co_source}:HS{float(ratio):.15g}:"
                            f"HG{float(hgas_pc):.15g}:CENTRAL_Q0_0P13"
                        )
                        summary, profile = base._build_cell(
                            config,
                            central,
                            maps,
                            cell_id=cell_id,
                            stellar_ml=stellar_ml,
                            co_source=co_source,
                            hstar_pc=rd_pc * float(ratio),
                            hgas_pc=float(hgas_pc),
                            cache=cache,
                        )
                        cells.append(summary)
                        profiles.append({"cell_id": cell_id, "radial_profile": profile})

    numerical = (
        (
            "COARSE_96_PRIMARY_PHYSICS",
            int(config["map_transform"]["coarse_grid_pixels"]),
            float(config["map_transform"]["primary_box_kpc"]),
            False,
        ),
        (
            "PADDED_384_PRIMARY_PHYSICS",
            int(config["map_transform"]["padded_grid_pixels"]),
            float(config["map_transform"]["padded_box_kpc"]),
            False,
        ),
        (
            "S4G_SIP_HEADER_SENSITIVITY_PRIMARY_PHYSICS",
            int(config["map_transform"]["primary_grid_pixels"]),
            float(config["map_transform"]["primary_box_kpc"]),
            True,
        ),
    )
    control_geometry: list[dict[str, Any]] = []
    for cell_id, n, box_kpc, use_sip in numerical:
        try:
            maps = _maps(config, central, images, n=n, box_kpc=box_kpc, use_sip=use_sip)
            summary, profile, control_rhalf, control_rd = _control_cell(
                config, central, maps, cell_id=cell_id
            )
        except NoConvergence:
            _require(use_sip, "non-SIP numerical control failed WCS convergence")
            summary = {
                "cell_id": cell_id,
                "object_id": central["object_id"],
                "status": "FAILED_RETAINED",
                "failure_code": "S4G_SIP_WCS_NONCONVERGENCE",
            }
            profile = []
            control_rhalf = None
            control_rd = None
        cells.append(summary)
        profiles.append(
            {
                "cell_id": cell_id,
                "status": summary.get("status", "PASS"),
                "radial_profile": profile,
            }
        )
        control_geometry.append(
            {
                "cell_id": cell_id,
                "metadata": central,
                "rhalf_pc": control_rhalf,
                "rd_pc": control_rd,
            }
        )

    for metadata in variants[1:]:
        cell_id = f"GEOMETRY_{metadata['geometry_variant_id']}_PRIMARY_PHYSICS"
        maps = _maps(config, metadata, images)
        summary, profile, control_rhalf, control_rd = _control_cell(
            config, metadata, maps, cell_id=cell_id
        )
        cells.append(summary)
        profiles.append({"cell_id": cell_id, "radial_profile": profile})
        control_geometry.append(
            {
                "cell_id": cell_id,
                "metadata": metadata,
                "rhalf_pc": control_rhalf,
                "rd_pc": control_rd,
            }
        )

    expected = (
        config["cell_contract"]["reliable_object_cells"]
        if geometry_row["orientation_flag"] == "ok"
        else config["cell_contract"]["uncertain_object_cells"]
    )
    _require(len(cells) == expected, "object cell count changed")
    middle_ratio = float(gravity["stellar_height_over_exponential_scale_cells"][1])
    primary_id = f"ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS{middle_ratio:.15g}:HG200:CENTRAL_Q0_0P13"
    primary = next(row for row in cells if row["cell_id"] == primary_id)
    coarse = next(row for row in cells if row["cell_id"] == "COARSE_96_PRIMARY_PHYSICS")
    padded = next(row for row in cells if row["cell_id"] == "PADDED_384_PRIMARY_PHYSICS")
    primary_g = float(primary["matched_acceleration"]["g_b_m_s2"])
    primary_potential = float(primary["matched_acceleration"]["potential_depth_c2"])
    convergence = {
        "coarse_g_relative": abs(float(coarse["matched_acceleration"]["g_b_m_s2"]) - primary_g)
        / max(primary_g, 1.0e-30),
        "padded_potential_relative": abs(
            float(padded["matched_acceleration"]["potential_depth_c2"]) - primary_potential
        )
        / max(primary_potential, 1.0e-30),
    }
    convergence["passed"] = bool(
        convergence["coarse_g_relative"] < 0.35 and convergence["padded_potential_relative"] < 0.35
    )
    _require(convergence["passed"], f"source convergence failed for {geometry_row['object_id']}")
    _require(1.0e7 < primary["stellar_mass_msun"] < 3.0e11, "stellar mass sanity failed")
    _require(1.0e6 < primary["hi_helium_mass_msun"] < 1.0e11, "HI mass sanity failed")
    _require(20.0 < primary["target_fwhm_pc"] < 3000.0, "beam sanity failed")
    return {
        "object_id": geometry_row["object_id"],
        "orientation_flag": geometry_row["orientation_flag"],
        "central_geometry": central,
        "rhalf_pc": rhalf_pc,
        "rd_pc": rd_pc,
        "primary_cell_id": primary_id,
        "primary_summary": primary,
        "convergence": convergence,
        "geometry_and_numerical_controls": control_geometry,
        "cell_summaries": cells,
        "cell_profiles": profiles,
        "cell_summary_root_sha256": content_sha256(cells),
        "cell_profile_root_sha256": content_sha256(profiles),
    }


def build_profiles(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    acquisition_receipt, geometry_config, operator_receipt = _load_dependencies(config)
    paths = _source_paths(config, acquisition_receipt)
    geometry = {row["object_id"]: row for row in geometry_config["objects"]}
    benchmark = base._benchmark_report(config)
    _require(all(benchmark["passed"].values()), "independent operator benchmark failed")
    objects = [
        _build_object(config, geometry[object_id], _load_images(object_id, paths))
        for object_id in _OBJECTS
    ]
    _require(sum(len(row["cell_summaries"]) for row in objects) == 393, "global cells changed")
    profiles: dict[str, Any] = {
        "schema": _PROFILE_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "source_acquisition_receipt_content_sha256": acquisition_receipt["content_sha256"],
        "geometry_source_content_sha256": config["geometry_binding"]["receipt_content_sha256"],
        "operator_benchmark_predecessor_content_sha256": operator_receipt["content_sha256"],
        "benchmarks": benchmark,
        "objects": objects,
        "cell_count": 393,
        "cell_summary_root_sha256": content_sha256([row["cell_summaries"] for row in objects]),
        "cell_profile_root_sha256": content_sha256([row["cell_profiles"] for row in objects]),
        "scientific_boundary": config["scientific_boundary"],
    }
    profiles["content_sha256"] = content_sha256(profiles)
    return profiles


def _public_object(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": row["object_id"],
        "orientation_flag": row["orientation_flag"],
        "central_geometry": row["central_geometry"],
        "rhalf_pc": row["rhalf_pc"],
        "rd_pc": row["rd_pc"],
        "primary_cell_id": row["primary_cell_id"],
        "primary_summary": row["primary_summary"],
        "convergence": row["convergence"],
        "geometry_and_numerical_controls": row["geometry_and_numerical_controls"],
        "cell_summary_root_sha256": row["cell_summary_root_sha256"],
        "cell_profile_root_sha256": row["cell_profile_root_sha256"],
    }


def build_packet(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = build_profiles(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "PASS_FIVE_REAL_SOURCE_MODEL_LIFTS_READY_FOR_FIXED_RESPONSE_TESTS",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "source_acquisition_binding": config["source_acquisition_binding"],
        "geometry_binding": config["geometry_binding"],
        "operator_benchmark_binding": config["operator_benchmark_binding"],
        "published_anchor_contract": config["published_anchor_contract"],
        "benchmarks": profiles["benchmarks"],
        "object_summaries": [_public_object(row) for row in profiles["objects"]],
        "cell_count": profiles["cell_count"],
        "cell_summary_root_sha256": profiles["cell_summary_root_sha256"],
        "cell_profile_root_sha256": profiles["cell_profile_root_sha256"],
        "private_profile_path": config["private_profile_output_path"],
        "private_profile_raw_sha256": hashlib.sha256(canonical_bytes(profiles)).hexdigest(),
        "private_profile_content_sha256": profiles["content_sha256"],
        "scientific_boundary": config["scientific_boundary"],
        "claims": config["claims"],
        "access_state": {
            "source_files_opened": 35,
            "source_bytes_opened": 90_449_926,
            "model_lifted_cells_derived": 393,
            "response_or_velocity_files_opened": 0,
            "response_rows_opened": 0,
            "scores_computed": 0,
            "models_fit": 0,
            "selection_events": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return profiles, receipt


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


def write_packet() -> str:
    config = load_config()
    profiles, receipt = build_packet(config)
    profile_status = _atomic_no_clobber(
        _repo_path(config["private_profile_output_path"]), canonical_bytes(profiles)
    )
    receipt_status = _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))
    return "CREATED" if "CREATED" in {profile_status, receipt_status} else "EXISTING_IDENTICAL"


def check_packet() -> str:
    config = load_config()
    profile_path = _repo_path(config["private_profile_output_path"])
    receipt_path = _repo_path(OUTPUT_PATH)
    _require(profile_path.is_file() and receipt_path.is_file(), "packet output missing")
    profiles, receipt = build_packet(config)
    _require(profile_path.read_bytes() == canonical_bytes(profiles), "profiles do not rebuild")
    _require(receipt_path.read_bytes() == canonical_bytes(receipt), "receipt does not rebuild")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_packet())
    elif args.command == "check":
        print(check_packet())
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
