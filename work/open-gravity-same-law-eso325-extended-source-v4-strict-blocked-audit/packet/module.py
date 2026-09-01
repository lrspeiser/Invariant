"""Frozen ESO 325-G004 source preflight and target-free same-state solver gates.

This module deliberately stops before scientific FITS arrays are decoded while the
registered reduction inputs in the V4 contract are incomplete.  Its numerical work is
entirely target-free and synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import irfftn, rfftn
from scipy.interpolate import RegularGridInterpolator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/open_gravity_same_law_eso325_extended_source_v4.json"
MODULE_PATH = Path(__file__).resolve()
TEST_PATH = REPOSITORY_ROOT / "tests/test_open_gravity_same_law_eso325_extended_source_v4.py"
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "runs/gravity/open-gravity-same-law-eso325-extended-source-v4/receipt.json"
)
ARTIFACT_DIRECTORY = OUTPUT_PATH.parent / "artifacts"


class SameLawESO325V4Error(RuntimeError):
    """Raised when a frozen source, seal, or numerical invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameLawESO325V4Error(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config["schema"] == "invariant-open-gravity-same-law-eso325-extended-source-4.0",
        "schema widened",
    )
    _require(config["status"] == "FROZEN_BEFORE_SCIENTIFIC_ARRAY_DECODE", "freeze lost")
    source = config["source_freeze"]
    _require(source["target"]["role"] == "DEVELOPMENT_ONLY", "ESO role widened")
    _require(source["scientific_arrays_decoded_at_freeze"] == 0, "array access at freeze")
    _require(source["scientific_response_rows_opened_at_freeze"] == 0, "response opened")
    roles = [row["role"] for row in source["exact_payloads"]]
    _require(
        roles
        == [
            "HST_F814W_LENS_LIGHT",
            "HST_F475W_ARC_PLUS_LENS_LIGHT",
            "MUSE_PRIMARY_CUBE",
            "MUSE_ARCHIVE_WHITELIGHT_AUXILIARY",
        ],
        "source roles changed",
    )
    for row in source["exact_payloads"] + source["paper_and_code"]:
        _require(len(row["sha256"]) == 64, f"invalid source hash: {row['role']}")
        _require(row["bytes"] > 0, f"invalid byte count: {row['role']}")
    reduction = config["reduction_contract"]
    _require(reduction["mode"].startswith("INDEPENDENT_REDUCTION"), "mode is not independent")
    _require("Do not fill" in reduction["fail_closed_rule"], "covariance fail-close lost")
    _require(reduction["muse"]["correction_to_prior_plan"].startswith("The paper reports 0.6"),
             "invented Voronoi plan returned")
    state = config["shared_physical_state"]
    _require(len(state["field_equations"]) == 2, "field equations incomplete")
    _require(len(state["metric_map"]) == 3, "metric map incomplete")
    forbidden = set(state["forbidden"])
    _require({"photon_multiplier", "lens_only_g", "lens_only_range"} <= forbidden,
             "photon prohibition weakened")
    _require(config["analysis_contract"]["primary_comparator"].startswith("GR_PLUS_STARS"),
             "GR+stars+NFW comparator lost")
    seal = config["slacs_confirmation_seal"]
    _require(seal["status"] == "SEALED_UNCHANGED", "SLACS seal opened")
    _require(seal["reserved_confirmation"] == 12, "SLACS confirmation count changed")
    _require(seal["confirmation_response_values_read_by_v4"] == 0, "SLACS response read")
    access = config["access_contract"]
    _require(access["development_targets"] == ["ESO 325-G004"], "target widened")
    _require(access["confirmation_targets_opened"] == 0, "confirmation target opened")
    _require(access["network_calls_by_builder"] == 0, "builder network widened")
    _require(access["model_calls"] == 0 and access["paid_calls"] == 0, "call budget widened")


def verify_frozen_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for item in config["source_freeze"]["exact_payloads"] + config["source_freeze"][
        "paper_and_code"
    ]:
        path = REPOSITORY_ROOT / item["path"]
        _require(path.is_file(), f"missing frozen source: {item['path']}")
        observed_bytes = path.stat().st_size
        observed_hash = file_sha256(path)
        _require(observed_bytes == item["bytes"], f"byte drift: {item['role']}")
        _require(observed_hash == item["sha256"], f"hash drift: {item['role']}")
        rows.append(
            {
                "role": item["role"],
                "path": item["path"],
                "bytes": observed_bytes,
                "sha256": observed_hash,
                "pass": True,
            }
        )
    prior = config["supersedes"]
    prior_path = REPOSITORY_ROOT / prior["path"]
    _require(file_sha256(prior_path) == prior["sha256"], "V3 receipt drift")
    seal_rows = []
    for name in ("sample_manifest", "predictor_manifest", "response_manifest"):
        item = config["slacs_confirmation_seal"][name]
        path = REPOSITORY_ROOT / item["path"]
        observed = file_sha256(path)
        _require(observed == item["sha256"], f"sealed manifest drift: {name}")
        seal_rows.append({"role": name, "path": item["path"], "sha256": observed})
    return {
        "all_exact_sources_pass": True,
        "source_rows": rows,
        "superseded_v3_sha256": prior["sha256"],
        "sealed_manifest_hashes_only": seal_rows,
        "sealed_response_manifest_deserialized": False,
    }


def _parse_fits_value(card: bytes) -> Any:
    text = card.decode("ascii", errors="replace")
    if len(text) < 10 or text[8] != "=":
        return None
    value = text[10:].split("/", 1)[0].strip()
    if value.startswith("'"):
        return value.strip().strip("'").strip()
    if value in {"T", "F"}:
        return value == "T"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value.replace("D", "E"))
        except ValueError:
            return value


def fits_structure(path: Path) -> dict[str, Any]:
    """Read FITS headers and seek over data; never decode a scientific array."""

    size = path.stat().st_size
    _require(size % 2880 == 0, f"FITS size is not block aligned: {path.name}")
    hdus: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        while handle.tell() < size:
            start = handle.tell()
            cards: list[bytes] = []
            found_end = False
            while not found_end:
                block = handle.read(2880)
                _require(len(block) == 2880, f"truncated FITS header: {path.name}")
                for offset in range(0, 2880, 80):
                    card = block[offset : offset + 80]
                    cards.append(card)
                    if card[:8].decode("ascii", errors="replace").strip() == "END":
                        found_end = True
                        break
            values: dict[str, Any] = {}
            for card in cards:
                key = card[:8].decode("ascii", errors="replace").strip()
                if key and key not in values:
                    values[key] = _parse_fits_value(card)
            naxis = int(values.get("NAXIS") or 0)
            axes = [int(values.get(f"NAXIS{axis}") or 0) for axis in range(1, naxis + 1)]
            bitpix = abs(int(values.get("BITPIX") or 8))
            pcount = int(values.get("PCOUNT") or 0)
            gcount = int(values.get("GCOUNT") or 1)
            data_bytes = (math.prod(axes) * bitpix // 8 + pcount) * gcount if axes else pcount
            padded = ((data_bytes + 2879) // 2880) * 2880
            next_hdu = handle.tell() + padded
            _require(next_hdu <= size, f"FITS HDU exceeds file: {path.name}")
            hdus.append(
                {
                    "index": len(hdus),
                    "offset": start,
                    "xtension": values.get("XTENSION", "PRIMARY"),
                    "extname": values.get("EXTNAME"),
                    "bitpix": int(values.get("BITPIX") or 8),
                    "axes": axes,
                    "data_bytes_skipped": data_bytes,
                    "checksum_keyword_present": "CHECKSUM" in values,
                    "datasum_keyword_present": "DATASUM" in values,
                }
            )
            handle.seek(next_hdu)
        final_position = handle.tell()
    _require(final_position == size, f"FITS trailing bytes: {path.name}")
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": size,
        "block_aligned": True,
        "hdu_count": len(hdus),
        "hdus": hdus,
        "scientific_array_elements_decoded": 0,
    }


def structural_source_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    structures = []
    for item in config["source_freeze"]["exact_payloads"]:
        structures.append(fits_structure(REPOSITORY_ROOT / item["path"]))
    return {
        "status": "PASS_EXACT_BYTES_AND_FITS_STRUCTURE_ONLY",
        "structures": structures,
        "fits_checksum_values_verified": False,
        "checksum_scope": "Keywords are inventoried structurally. Payload checksums require reading array bytes and remain after the pre-decode source gate.",
        "scientific_arrays_decoded": 0,
        "scientific_response_rows_opened": 0,
    }


@dataclass(frozen=True)
class ExtendedState:
    coordinates: np.ndarray
    cell: float
    density: np.ndarray
    U: np.ndarray
    Y: np.ndarray
    Phi: np.ndarray
    Psi: np.ndarray


def synthetic_density(n: int, physical_extent: float) -> tuple[np.ndarray, np.ndarray, float]:
    cell = physical_extent / n
    coordinates = (np.arange(n, dtype=float) - (n - 1) / 2) * cell
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    components = (
        (0.60, (0.25, -0.18, 0.10), (0.72, 1.00, 0.62)),
        (0.25, (-0.55, 0.36, -0.12), (0.48, 0.65, 0.42)),
        (0.15, (0.00, 0.00, 0.00), (2.20, 2.20, 2.20)),
    )
    density = np.zeros((n, n, n), dtype=float)
    for mass, centre, sigma in components:
        exponent = (
            ((x - centre[0]) / sigma[0]) ** 2
            + ((y - centre[1]) / sigma[1]) ** 2
            + ((z - centre[2]) / sigma[2]) ** 2
        )
        raw = np.exp(-0.5 * exponent)
        density += mass * raw / (raw.sum() * cell**3)
    return coordinates, density, cell


def solve_extended_state(
    density: np.ndarray,
    coordinates: np.ndarray,
    cell: float,
    *,
    g: float,
    range_kpc: float,
) -> ExtendedState:
    """Solve both potentials from one density; no observable-specific parameter is accepted."""

    _require(density.ndim == 3 and len(set(density.shape)) == 1, "density grid must be cubic")
    _require(np.all(np.isfinite(density)) and np.all(density >= 0), "density invalid")
    _require(range_kpc > 0, "range must be positive")
    n = density.shape[0]
    padded_n = 2 * n
    start = (padded_n - n) // 2
    stop = start + n
    padded = np.zeros((padded_n, padded_n, padded_n), dtype=float)
    padded[start:stop, start:stop, start:stop] = density
    transformed = rfftn(padded)
    wave = 2.0 * np.pi * np.fft.fftfreq(padded_n, d=cell)
    wave_z = 2.0 * np.pi * np.fft.rfftfreq(padded_n, d=cell)
    k2 = wave[:, None, None] ** 2 + wave[None, :, None] ** 2 + wave_z[None, None, :] ** 2
    newton_kernel = np.zeros_like(k2)
    nonzero = k2 > 0
    newton_kernel[nonzero] = -4.0 * np.pi / k2[nonzero]
    mu2 = 1.0 / range_kpc**2
    yukawa_kernel = -4.0 * np.pi / (k2 + mu2)
    yukawa_kernel[0, 0, 0] = 0.0
    U_full = irfftn(transformed * newton_kernel, s=padded.shape).real
    Y_full = irfftn(transformed * yukawa_kernel, s=padded.shape).real
    U = U_full[start:stop, start:stop, start:stop]
    Y = Y_full[start:stop, start:stop, start:stop]
    phi = U + (4.0 / 3.0) * g * Y
    psi = U + (2.0 / 3.0) * g * Y
    for array in (density, U, Y, phi, psi):
        array.setflags(write=False)
    coordinates.setflags(write=False)
    return ExtendedState(coordinates, cell, density, U, Y, phi, psi)


def matter_observable(state: ExtendedState, points: np.ndarray) -> np.ndarray:
    """Return acceleration vectors from Phi at fixed points."""

    gradients = np.gradient(state.Phi, state.cell, edge_order=2)
    values = []
    for component in gradients:
        interpolator = RegularGridInterpolator(
            (state.coordinates,) * 3, -component, bounds_error=True
        )
        values.append(interpolator(points))
    return np.stack(values, axis=1)


def photon_observable(state: ExtendedState, points_xy: np.ndarray) -> np.ndarray:
    """Return transverse deflection from Phi+Psi; it has no photon parameter."""

    combined = state.Phi + state.Psi
    gx, gy, _ = np.gradient(combined, state.cell, edge_order=2)
    projected_x = np.sum(gx, axis=2) * state.cell / 100.0
    projected_y = np.sum(gy, axis=2) * state.cell / 100.0
    axes = (state.coordinates, state.coordinates)
    ax = RegularGridInterpolator(axes, projected_x, bounds_error=True)(points_xy)
    ay = RegularGridInterpolator(axes, projected_y, bounds_error=True)(points_xy)
    return np.stack((ax, ay), axis=1)


def extended_source_image(state: ExtendedState, image_coordinates: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(image_coordinates, image_coordinates, indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel()), axis=1)
    deflection = photon_observable(state, points)
    beta = points - deflection
    clumps = (
        (1.0, -0.25, 0.10, 0.28),
        (0.65, 0.38, -0.22, 0.18),
        (0.40, 0.05, 0.48, 0.13),
    )
    brightness = np.zeros(len(points), dtype=float)
    for amplitude, bx, by, sigma in clumps:
        radius2 = (beta[:, 0] - bx) ** 2 + (beta[:, 1] - by) ** 2
        brightness += amplitude * np.exp(-0.5 * radius2 / sigma**2)
    return brightness.reshape(xx.shape)


def _normalized_residual(truth: np.ndarray, prediction: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(truth)), 1e-15)
    return float(np.linalg.norm(truth - prediction) / scale)


def target_free_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["target_free_gate"]
    n = int(gate["grid"]["cells_per_axis"])
    cell = float(gate["grid"]["cell_kpc"])
    extent = n * cell
    injected = gate["injected_parameters"]
    coordinates, density, observed_cell = synthetic_density(n, extent)
    truth = solve_extended_state(
        density,
        coordinates,
        observed_cell,
        g=float(injected["g"]),
        range_kpc=float(injected["range_kpc"]),
    )
    gr = solve_extended_state(
        density, coordinates.copy(), observed_cell, g=0.0, range_kpc=float(injected["range_kpc"])
    )
    short_range = solve_extended_state(
        density,
        coordinates.copy(),
        observed_cell,
        g=float(injected["g"]),
        range_kpc=1e-4,
    )
    points = np.array(
        [[0.8, 0.0, 0.0], [1.2, 0.3, 0.0], [1.8, -0.4, 0.2], [2.4, 0.5, -0.3]],
        dtype=float,
    )
    points_xy = points[:, :2]
    matter_truth = matter_observable(truth, points)
    matter_gr = matter_observable(gr, points)
    lens_truth = photon_observable(truth, points_xy)
    lens_gr = photon_observable(gr, points_xy)
    image_axis = np.linspace(-2.8, 2.8, 65)
    image_truth = extended_source_image(truth, image_axis)
    image_gr = extended_source_image(gr, image_axis)

    # At fixed range the state is linear in g, so this is an independent joint recovery.
    unit = solve_extended_state(
        density, coordinates.copy(), observed_cell, g=1.0, range_kpc=float(injected["range_kpc"])
    )
    matter_basis = matter_observable(unit, points) - matter_gr
    lens_basis = photon_observable(unit, points_xy) - lens_gr
    joint_truth = np.concatenate(((matter_truth - matter_gr).ravel(), (lens_truth - lens_gr).ravel()))
    joint_basis = np.concatenate((matter_basis.ravel(), lens_basis.ravel()))
    recovered_g = float(np.dot(joint_basis, joint_truth) / np.dot(joint_basis, joint_basis))

    n_reference = 49
    ref_coordinates, ref_density, ref_cell = synthetic_density(n_reference, extent)
    reference = solve_extended_state(
        ref_density,
        ref_coordinates,
        ref_cell,
        g=float(injected["g"]),
        range_kpc=float(injected["range_kpc"]),
    )
    reference_vector = np.concatenate(
        (matter_observable(reference, points).ravel(), photon_observable(reference, points_xy).ravel())
    )
    primary_vector = np.concatenate((matter_truth.ravel(), lens_truth.ravel()))
    convergence_relative_rms = float(
        np.linalg.norm(primary_vector - reference_vector) / np.linalg.norm(primary_vector)
    )

    # A separate spherical fixture makes reflection/permutation symmetries exact expectations.
    sym_n = 33
    sym_extent = 8.0
    sym_coordinates = (np.arange(sym_n) - (sym_n - 1) / 2) * (sym_extent / sym_n)
    sx, sy, sz = np.meshgrid(sym_coordinates, sym_coordinates, sym_coordinates, indexing="ij")
    sym_density = np.exp(-0.5 * (sx**2 + sy**2 + sz**2) / 0.9**2)
    sym_density /= sym_density.sum() * (sym_extent / sym_n) ** 3
    sym_state = solve_extended_state(
        sym_density,
        sym_coordinates.copy(),
        sym_extent / sym_n,
        g=float(injected["g"]),
        range_kpc=float(injected["range_kpc"]),
    )
    denom = max(float(np.max(np.abs(sym_state.Phi))), 1e-15)
    reflection_error = float(np.max(np.abs(sym_state.Phi - sym_state.Phi[::-1])) / denom)
    permutation_error = float(np.max(np.abs(sym_state.Phi - np.swapaxes(sym_state.Phi, 0, 1))) / denom)

    mass = float(density.sum() * observed_cell**3)
    mass_error = abs(mass - 1.0)
    gr_state_error = max(
        float(np.max(np.abs(gr.Phi - gr.U))),
        float(np.max(np.abs(gr.Psi - gr.U))),
    )
    short_range_error = max(
        float(np.linalg.norm(short_range.Phi - gr.Phi) / np.linalg.norm(gr.Phi)),
        float(np.linalg.norm(short_range.Psi - gr.Psi) / np.linalg.norm(gr.Psi)),
    )
    matter_gr_residual = _normalized_residual(matter_truth, matter_gr)
    lens_gr_residual = _normalized_residual(lens_truth, lens_gr)
    image_gr_residual = _normalized_residual(image_truth, image_gr)

    # One mass scale cannot generally absorb different Phi and Phi+Psi responses.
    gr_joint = np.concatenate((matter_gr.ravel(), lens_gr.ravel()))
    truth_joint = np.concatenate((matter_truth.ravel(), lens_truth.ravel()))
    mass_scale = float(np.dot(gr_joint, truth_joint) / np.dot(gr_joint, gr_joint))
    mass_rescale_residual = _normalized_residual(truth_joint, mass_scale * gr_joint)

    public_parameters = {
        "matter_observable": list(inspect.signature(matter_observable).parameters),
        "photon_observable": list(inspect.signature(photon_observable).parameters),
        "extended_source_image": list(inspect.signature(extended_source_image).parameters),
    }
    forbidden_parameter_names = {
        "g",
        "range",
        "range_kpc",
        "coupling",
        "photon_coupling",
        "photon_multiplier",
        "lens_multiplier",
    }
    no_photon_knob = all(
        name.lower() not in forbidden_parameter_names
        for parameters in public_parameters.values()
        for name in parameters
    )
    rows = {
        "nonnegative_density": bool(np.all(density >= 0)),
        "relative_mass_error": mass_error,
        "gr_state_max_absolute_error": gr_state_error,
        "short_range_relative_state_error": short_range_error,
        "reflection_relative_error": reflection_error,
        "axis_permutation_relative_error": permutation_error,
        "grid_convergence_relative_rms": convergence_relative_rms,
        "recovered_g_at_frozen_range": recovered_g,
        "recovered_g_absolute_error": abs(recovered_g - float(injected["g"])),
        "matter_gr_relative_residual": matter_gr_residual,
        "lensing_gr_relative_residual": lens_gr_residual,
        "extended_image_gr_relative_residual": image_gr_residual,
        "best_common_mass_rescale": mass_scale,
        "common_mass_rescale_joint_relative_residual": mass_rescale_residual,
        "candidate_noiseless_joint_relative_residual": 0.0,
        "public_route_parameters": public_parameters,
        "no_photon_only_parameter": no_photon_knob,
    }
    pass_gate = (
        rows["nonnegative_density"]
        and mass_error <= 1e-12
        and gr_state_error == 0.0
        and short_range_error <= 1e-6
        and reflection_error <= 5e-10
        and permutation_error <= 5e-10
        and convergence_relative_rms <= 0.08
        and rows["recovered_g_absolute_error"] <= 1e-10
        and matter_gr_residual > 1e-4
        and lens_gr_residual > 1e-4
        and image_gr_residual > 1e-6
        and mass_rescale_residual > 1e-5
        and no_photon_knob
    )
    return {
        "status": "PASS_TARGET_FREE_SHARED_STATE" if pass_gate else "FAIL_TARGET_FREE_SHARED_STATE",
        "pass": pass_gate,
        "grid": {"primary_cells": n, "reference_cells": n_reference, "extent_kpc": extent},
        "injection": injected,
        "metrics": rows,
        "retained_countermodels": {
            "GR": {
                "matter_relative_residual": matter_gr_residual,
                "lensing_relative_residual": lens_gr_residual,
                "extended_image_relative_residual": image_gr_residual,
            },
            "COMMON_MASS_RESCALE": {
                "best_scale": mass_scale,
                "joint_relative_residual": mass_rescale_residual,
            },
            "SPLIT_STATE": {
                "status": "FORBIDDEN_IDENTIFIABILITY_CONTROL_RETAINED_NOT_ADMITTED",
                "explanation": "A lens-only coefficient could fit the synthetic lens route independently but violates the one-state contract and is never an eligible candidate.",
            },
        },
        "claim_boundary": "A target-free implementation and limit test only; it is not evidence about ESO 325-G004 or gravity.",
    }


def source_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = list(config["reduction_contract"]["missing_public_inputs_before_decode"])
    return {
        "status": "BLOCK_BEFORE_SCIENTIFIC_ARRAY_DECODE",
        "reason": "The exact raw archive bytes are local, but an honest independent joint likelihood still lacks hash-bound stellar templates and pre-array evidence that the frozen empirical PSF, astrometry, LSF and covariance gates can be met.",
        "missing_or_unverified": missing,
        "paper_registered_products_publicly_located": False,
        "paper_kinematic_bins_are_voronoi": False,
        "paper_kinematic_sampling": "0.6-arcsec square spatial pixels",
        "scientific_arrays_decoded": 0,
        "eso_development_score_computed": False,
        "slacs_confirmation_opened": False,
    }


def build_artifacts(config: Mapping[str, Any]) -> dict[str, bytes]:
    sources = verify_frozen_sources(config)
    structures = structural_source_preflight(config)
    gate = target_free_gate(config)
    readiness = source_readiness(config)
    report = (
        "# Lane 7 V4: ESO 325-G004 extended-source preflight\n\n"
        f"- Exact source bytes: **{'PASS' if sources['all_exact_sources_pass'] else 'FAIL'}**.\n"
        f"- FITS structure without array decode: **{structures['status']}**.\n"
        f"- Target-free one-state solver: **{gate['status']}**.\n"
        f"- Real ESO score: **{readiness['status']}**.\n"
        "- SLACS confirmation: **SEALED; zero response values opened**.\n\n"
        "The public paper supplement corrects the earlier Voronoi assumption: its MUSE analysis used "
        "0.6-arcsec square pixels. The archive contains the raw/reduced HST and MUSE products, but not "
        "the registered paper mask, PSFs, kinematic table/covariance, posterior, or joint likelihood. "
        "V4 therefore freezes an independent reduction and refuses to invent covariance. The synthetic "
        "shared-state gate can pass without granting permission to decode or score ESO arrays.\n"
    ).encode()
    return {
        "exact-source-receipt.json": _json_bytes(sources),
        "fits-structure-only.json": _json_bytes(structures),
        "target-free-shared-state-gate.json": _json_bytes(gate),
        "source-readiness.json": _json_bytes(readiness),
        "frozen-reduction-and-analysis-contract.json": _json_bytes(
            {
                "reduction_contract": config["reduction_contract"],
                "cosmology_and_units": config["cosmology_and_units"],
                "shared_physical_state": config["shared_physical_state"],
                "analysis_contract": config["analysis_contract"],
                "slacs_confirmation_seal": config["slacs_confirmation_seal"],
            }
        ),
        "report.md": report,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    artifacts = build_artifacts(config)
    source = json.loads(artifacts["exact-source-receipt.json"])
    structure = json.loads(artifacts["fits-structure-only.json"])
    gate = json.loads(artifacts["target-free-shared-state-gate.json"])
    readiness = json.loads(artifacts["source-readiness.json"])
    _require(source["all_exact_sources_pass"] is True, "source gate failed")
    _require(gate["pass"] is True, "target-free gate failed")
    _require(readiness["status"] == "BLOCK_BEFORE_SCIENTIFIC_ARRAY_DECODE", "source widened")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-same-law-eso325-extended-source-receipt-4.0",
        "package_id": config["package_id"],
        "status": "SOURCE_BLOCKED_ESO_REDUCTION_INPUTS_AFTER_TARGET_FREE_PASS",
        "decision": "KEEP_ESO_SCIENTIFIC_ARRAYS_AND_ALL_SLACS_CONFIRMATION_RESPONSES_SEALED",
        "exact_source_status": source,
        "fits_structure_status": structure,
        "target_free_gate": gate,
        "source_readiness": readiness,
        "source_anomalies_retained": config["source_freeze"]["source_anomalies_retained"],
        "claim_boundary": {
            "establishes": [
                "four exact local ESO/HST/MUSE payload hashes",
                "an append-only reduction and analysis contract frozen before array decode",
                "a target-free extended-source one-state matter-and-lensing solver passes its registered limits",
                "the prior Voronoi assumption is corrected to the paper's 0.6-arcsec square pixels",
            ],
            "does_not_establish": [
                "a reduction, fit, or score for ESO 325-G004",
                "reproduction of Collett et al. 2018",
                "evidence for the Yukawa comparator or any modified gravity",
                "a healthy nonlinear completion",
                "any result on the 12 sealed SLACS confirmation lenses",
            ],
        },
        "access_accounting": {
            "scientific_fits_array_elements_decoded": 0,
            "eso_scientific_response_rows_opened": 0,
            "eso_scores_computed": 0,
            "slacs_confirmation_response_values_opened": 0,
            "network_calls_by_builder": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "post_response_tuning_calls": 0,
        },
        "artifact_manifest": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(artifacts.items())
        },
        "artifact_bindings": {
            "config": {"path": CONFIG_PATH.relative_to(REPOSITORY_ROOT).as_posix(), "sha256": file_sha256(CONFIG_PATH)},
            "module": {"path": MODULE_PATH.relative_to(REPOSITORY_ROOT).as_posix(), "sha256": file_sha256(MODULE_PATH)},
            "test": {"path": TEST_PATH.relative_to(REPOSITORY_ROOT).as_posix(), "sha256": file_sha256(TEST_PATH)},
        },
    }
    receipt["content_sha256"] = _content_sha256(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"existing artifact differs: {path}")
        return "EXISTING_IDENTICAL"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        _require(path.read_bytes() == payload, f"concurrent artifact differs: {path}")
        return "EXISTING_IDENTICAL"
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def write_packet() -> str:
    config = load_config()
    statuses = [
        _atomic_no_clobber(ARTIFACT_DIRECTORY / name, payload)
        for name, payload in build_artifacts(config).items()
    ]
    statuses.append(_atomic_no_clobber(OUTPUT_PATH, _json_bytes(build_receipt())))
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def validate_receipt() -> None:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(observed == build_receipt(), "receipt differs from deterministic rebuild")
    for name, payload in build_artifacts(load_config()).items():
        _require((ARTIFACT_DIRECTORY / name).read_bytes() == payload, f"artifact drift: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "status"))
    arguments = parser.parse_args(argv)
    if arguments.action == "build":
        print(write_packet())
    elif arguments.action == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt()
        print(receipt["status"])
        print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
