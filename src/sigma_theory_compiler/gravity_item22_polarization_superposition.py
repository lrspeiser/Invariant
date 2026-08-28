"""Frozen Item 22 scalar/vector/tensor superposition search.

The sample is selected from response-blind lens identities, redshifts, and independent
Legacy Survey DR9 Tractor photometry.  Only exploration-role velocity dispersions and
Einstein radii may be acquired after both the science and sample freezes are bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import subprocess
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    _angular_diameter_distances,
    _parse_vizier_tsv,
    hernquist_projected_mass_fraction,
)

CONFIG_PATH = Path("configs/gravity_item22_polarization_superposition_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item22_polarization_superposition.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem22Error(RuntimeError):
    """Raised when an Item 22 freeze, leakage, or replay invariant is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _content_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_sha256", None)
    result["content_sha256"] = _sha256_bytes(_canonical_bytes(result))
    return result


def _verify_content_hash(payload: Mapping[str, Any], label: str) -> None:
    expected = payload.get("content_sha256")
    if not isinstance(expected, str):
        raise GravityItem22Error(f"{label} has no content hash")
    body = dict(payload)
    body.pop("content_sha256", None)
    if _sha256_bytes(_canonical_bytes(body)) != expected:
        raise GravityItem22Error(f"{label} content hash changed")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityItem22Error(f"expected JSON object: {path}")
    return value


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem22Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem22Error(f"{label} is not an ancestor of HEAD")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item22-polarization-superposition-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 22:
        raise GravityItem22Error("unexpected Item 22 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem22Error("stable gravity goal changed")
    generator = config["candidate_generator"]
    if int(generator["candidate_cells"]) != 262144:
        raise GravityItem22Error("candidate boundary changed")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem22Error("post-response candidates entered the contract")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem22Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem22Error("paid calls are outside Item 22")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem22Error("equal-viability policy changed")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem22Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem22Error("could not read frozen module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem22Error("Item 22 module differs from scientific freeze")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "identity_catalog",
        "legacy_predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "exploration_responses",
        "response_source_manifest",
        "compute_manifest",
    )
    return {key: base / str(config["paths"][key]) for key in keys}


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in (
        "identity_catalog",
        "legacy_predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        repo_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repo_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem22Error(f"{key} differs from sample freeze")


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item22/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        body = response.read()
        headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    if not body:
        raise GravityItem22Error(f"empty source response: {url}")
    return body, headers


def _format_float(value: float) -> str:
    return f"{float(value):.12e}"


def _normal_identity(value: str) -> str:
    answer = "".join(c for c in str(value).upper() if c.isalnum() or c in "+-")
    return answer.replace("SDSS", "").replace("SL2S", "")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").strip()
    return str(value).strip()


def _angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = math.radians(ra2 - ra1)
    de1, de2 = math.radians(dec1), math.radians(dec2)
    cosine = math.sin(de1) * math.sin(de2) + math.cos(de1) * math.cos(de2) * math.cos(dra)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600.0


def _legacy_url(config: Mapping[str, Any], ra: float, dec: float) -> str:
    width = float(config["sources"]["legacy_half_width_deg"])
    params = {
        "ralo": f"{ra - width:.8f}",
        "rahi": f"{ra + width:.8f}",
        "declo": f"{dec - width:.8f}",
        "dechi": f"{dec + width:.8f}",
    }
    return str(config["sources"]["legacy_catalog_service"]) + "?" + urllib.parse.urlencode(params)


def _prior_identities(root: Path, config: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for relative in config["sources"]["predecessor_sample_manifests"]:
        manifest = _read_json(root / str(relative))
        for row in manifest.get("objects", []):
            if isinstance(row, Mapping) and row.get("name"):
                found.add(_normal_identity(str(row["name"])))
    return found


def _nearest_legacy_row(
    body: bytes, ra: float, dec: float, config: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        from astropy.io import fits

        with fits.open(io.BytesIO(body), memmap=False) as hdul:
            table = hdul[1].data
            rows = [row for row in table]
    except (OSError, IndexError, TypeError, ValueError) as error:  # pragma: no cover
        return None, [f"catalog_parse:{type(error).__name__}"]
    if not rows:
        return None, ["no_catalog_rows"]
    ranked = [
        (
            _angular_separation_arcsec(ra, dec, float(row["ra"]), float(row["dec"])),
            row,
        )
        for row in rows
    ]
    separation, row = min(ranked, key=lambda value: value[0])
    fields: dict[str, Any] = {
        "release": int(row["release"]),
        "brickid": int(row["brickid"]),
        "brickname": _decode(row["brickname"]),
        "objid": int(row["objid"]),
        "brick_primary": bool(row["brick_primary"]),
        "type": _decode(row["type"]),
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
        "center_separation_arcsec": separation,
        "shape_r": float(row["shape_r"]),
        "shape_r_ivar": float(row["shape_r_ivar"]),
        "shape_e1": float(row["shape_e1"]),
        "shape_e2": float(row["shape_e2"]),
        "sersic": float(row["sersic"]),
        "maskbits": int(row["maskbits"]),
        "fitbits": int(row["fitbits"]),
    }
    for band in ("g", "r", "z"):
        for prefix in (
            "flux",
            "flux_ivar",
            "mw_transmission",
            "nobs",
            "fracflux",
            "fracmasked",
            "fracin",
            "allmask",
            "rchisq",
        ):
            value = row[f"{prefix}_{band}"]
            fields[f"{prefix}_{band}"] = int(value) if prefix in ("nobs", "allmask") else float(value)
    quality = config["predictor_quality"]
    failures: list[str] = []
    if separation > float(quality["maximum_center_separation_arcsec"]):
        failures.append("center_separation")
    if fields["type"] not in quality["allowed_morphology_types"]:
        failures.append("morphology")
    if bool(quality["require_brick_primary"]) and not fields["brick_primary"]:
        failures.append("brick_primary")
    if fields["shape_r"] < float(quality["minimum_shape_radius_arcsec"]):
        failures.append("shape_radius")
    shape_snr = fields["shape_r"] * math.sqrt(max(fields["shape_r_ivar"], 0.0))
    if shape_snr < float(quality["minimum_shape_signal_to_noise"]):
        failures.append("shape_signal_to_noise")
    for band in ("g", "r", "z"):
        flux = fields[f"flux_{band}"]
        ivar = fields[f"flux_ivar_{band}"]
        if flux <= 0 or ivar <= 0 or flux * math.sqrt(ivar) < float(
            quality["minimum_flux_signal_to_noise_each_grz"]
        ):
            failures.append(f"flux_signal_to_noise_{band}")
        if fields[f"nobs_{band}"] <= 0:
            failures.append(f"nobs_{band}")
        if fields[f"fracmasked_{band}"] > float(quality["maximum_fracmasked_each_grz"]):
            failures.append(f"fracmasked_{band}")
        if fields[f"fracflux_{band}"] > float(quality["maximum_fracflux_each_grz"]):
            failures.append(f"fracflux_{band}")
        if fields[f"fracin_{band}"] < float(quality["minimum_fracin_each_grz"]):
            failures.append(f"fracin_{band}")
        if bool(quality["require_allmask_zero_each_grz"]) and fields[f"allmask_{band}"] != 0:
            failures.append(f"allmask_{band}")
        if bool(quality["require_positive_mw_transmission_each_grz"]) and fields[
            f"mw_transmission_{band}"
        ] <= 0:
            failures.append(f"mw_transmission_{band}")
    fields["shape_signal_to_noise"] = shape_snr
    return fields, sorted(set(failures))


def _derived_predictor(
    identity: Mapping[str, Any], tractor: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    z_lens = float(identity["zl"])
    z_source = float(identity["zs"])
    d_lens, _, _ = _angular_diameter_distances(z_lens, z_source, config)
    d_luminosity_kpc = d_lens * (1.0 + z_lens) ** 2
    distance_modulus = 5.0 * math.log10(d_luminosity_kpc * 100.0)
    magnitudes: dict[str, float] = {}
    for band in ("g", "r", "z"):
        corrected_flux = float(tractor[f"flux_{band}"]) / float(
            tractor[f"mw_transmission_{band}"]
        )
        magnitudes[band] = 22.5 - 2.5 * math.log10(corrected_flux)
    absolute_z = magnitudes["z"] - distance_modulus + 2.5 * math.log10(1.0 + z_lens)
    luminosity = 10.0 ** (
        -0.4 * (absolute_z - float(config["physics"]["constants"]["absolute_z_solar_AB"]))
    )
    reff_kpc = (
        float(tractor["shape_r"])
        * float(config["physics"]["constants"]["arcsec_to_radian"])
        * d_lens
    )
    ellipticity = min(math.hypot(float(tractor["shape_e1"]), float(tractor["shape_e2"])), 0.999)
    axis_ratio = (1.0 - ellipticity) / (1.0 + ellipticity)
    return {
        "name": str(identity["Name"]).strip(),
        "normalized_identity": _normal_identity(str(identity["Name"])),
        "survey": str(identity["Survey"]).strip(),
        "z_lens": z_lens,
        "z_source": z_source,
        "catalog_ra_deg": float(identity["_RA"]),
        "catalog_dec_deg": float(identity["_DE"]),
        "tractor_release": int(tractor["release"]),
        "tractor_brickid": int(tractor["brickid"]),
        "tractor_brickname": str(tractor["brickname"]),
        "tractor_objid": int(tractor["objid"]),
        "tractor_type": str(tractor["type"]),
        "tractor_ra_deg": float(tractor["ra"]),
        "tractor_dec_deg": float(tractor["dec"]),
        "center_separation_arcsec": float(tractor["center_separation_arcsec"]),
        "reff_arcsec": float(tractor["shape_r"]),
        "reff_kpc": reff_kpc,
        "axis_ratio": axis_ratio,
        "sersic": float(tractor["sersic"]),
        "z_luminosity_Lsun": luminosity,
        "g_minus_r": magnitudes["g"] - magnitudes["r"],
        "r_minus_z": magnitudes["r"] - magnitudes["z"],
        "max_fracflux_grz": max(float(tractor[f"fracflux_{b}"]) for b in ("g", "r", "z")),
        "max_fracmasked_grz": max(
            float(tractor[f"fracmasked_{b}"]) for b in ("g", "r", "z")
        ),
        "shape_signal_to_noise": float(tractor["shape_signal_to_noise"]),
        "minimum_flux_signal_to_noise_grz": min(
            float(tractor[f"flux_{b}"]) * math.sqrt(float(tractor[f"flux_ivar_{b}"]))
            for b in ("g", "r", "z")
        ),
    }


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _format_float(value) if isinstance(value, float) else value
                    for key, value in row.items()
                    if key in columns
                }
            )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _build_sample(
    predictors: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    sample = config["sample"]
    ranked = sorted(
        (dict(row) for row in predictors),
        key=lambda row: _hmac_rank(str(sample["selection_key"]), str(row["name"])),
    )
    selected = ranked[: int(sample["selected_predictor_count"])]
    selected.sort(key=lambda row: (float(row["z_lens"]), str(row["name"])))
    width = int(sample["objects_per_redshift_stratum"])
    for index, row in enumerate(selected):
        row["redshift_stratum"] = index // width
    for stratum in range(int(sample["redshift_strata"])):
        group = [row for row in selected if int(row["redshift_stratum"]) == stratum]
        if len(group) != width:
            raise GravityItem22Error("redshift stratum changed size")
        group.sort(key=lambda row: _hmac_rank(str(sample["role_key"]), str(row["name"])))
        held = {
            str(row["name"])
            for row in group[: int(sample["confirmation_per_stratum"])]
        }
        for row in group:
            row["role"] = "confirmation" if str(row["name"]) in held else "exploration"
    exploration = [row for row in selected if row["role"] == "exploration"]
    exploration.sort(key=lambda row: _hmac_rank(str(sample["fold_key"]), str(row["name"])))
    for index, row in enumerate(exploration):
        row["fold"] = index % int(sample["outer_folds"])
    for row in selected:
        if row["role"] == "confirmation":
            row["fold"] = None
    selected.sort(key=lambda row: str(row["name"]))
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item22-polarization-sample-1.0",
            "selection_used_response_values": False,
            "response_values_read": 0,
            "confirmation_opened": False,
            "counts": {
                "predictor_quality_eligible": len(predictors),
                "selected": len(selected),
                "exploration": len(exploration),
                "confirmation": len(selected) - len(exploration),
            },
            "fold_counts": dict(sorted(Counter(str(row["fold"]) for row in exploration).items())),
            "objects": selected,
        }
    )


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    niche = np.repeat(np.arange(4, dtype=np.int16), count // 4)
    polarity = np.tile(
        np.repeat(np.asarray([0, 1], dtype=np.int16), count // 8), 4
    )
    joint_permutation = random.permutation(count)
    niche = niche[joint_permutation]
    polarity = polarity[joint_permutation]
    ordered_distinct_ranges = np.asarray(
        [(a, b) for a in range(4) for b in range(4) if a != b], dtype=np.int16
    )
    range_pairs = ordered_distinct_ranges[
        random.integers(0, len(ordered_distinct_ranges), count)
    ]
    return {
        "niche": niche,
        "polarity": polarity,
        "amplitude": random.integers(
            0, len(generator["amplitude_magnitudes"]), count, dtype=np.int16
        ),
        "lambda_scalar": random.integers(
            0, len(generator["lambda_scalar_kpc"]), count, dtype=np.int16
        ),
        "range_vector": range_pairs[:, 0].copy(),
        "range_tensor": range_pairs[:, 1].copy(),
        "pure_mode": random.integers(0, 3, count, dtype=np.int16),
        "pair_mode": random.integers(0, 3, count, dtype=np.int16),
        "pair_mixing": random.integers(
            0, len(generator["pair_mixing_fractions"]), count, dtype=np.int16
        ),
        "triple_simplex": random.integers(
            0, len(generator["triple_simplex_weights"]), count, dtype=np.int16
        ),
        "interaction_pair": random.integers(0, 3, count, dtype=np.int16),
        "interaction_fraction": random.integers(
            0, len(generator["interaction_fractions"]), count, dtype=np.int16
        ),
        "interaction_phase": random.integers(0, 2, count, dtype=np.int16),
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    g = config["candidate_generator"]
    return {
        "niche": xp.asarray(arrays["niche"][begin:end]),
        "polarity": xp.asarray(np.asarray(g["polarities"])[arrays["polarity"][begin:end]]),
        "amplitude": xp.asarray(
            np.asarray(g["amplitude_magnitudes"])[arrays["amplitude"][begin:end]]
        ),
        "lambda_scalar": xp.asarray(
            np.asarray(g["lambda_scalar_kpc"])[arrays["lambda_scalar"][begin:end]]
        ),
        "range_vector": xp.asarray(
            np.asarray(g["range_ratios"])[arrays["range_vector"][begin:end]]
        ),
        "range_tensor": xp.asarray(
            np.asarray(g["range_ratios"])[arrays["range_tensor"][begin:end]]
        ),
        "pure_mode": xp.asarray(arrays["pure_mode"][begin:end]),
        "pair_mode": xp.asarray(arrays["pair_mode"][begin:end]),
        "pair_mixing": xp.asarray(
            np.asarray(g["pair_mixing_fractions"])[arrays["pair_mixing"][begin:end]]
        ),
        "triple_simplex": xp.asarray(
            np.asarray(g["triple_simplex_weights"])[arrays["triple_simplex"][begin:end]]
        ),
        "interaction_pair": xp.asarray(arrays["interaction_pair"][begin:end]),
        "interaction_fraction": xp.asarray(
            np.asarray(g["interaction_fractions"])[arrays["interaction_fraction"][begin:end]]
        ),
        "interaction_phase": xp.asarray(
            np.asarray(g["interaction_phases"])[arrays["interaction_phase"][begin:end]]
        ),
    }


def _mode_weights(values: Mapping[str, Any], xp: Any) -> Any:
    count = int(values["niche"].shape[0])
    pure = xp.zeros((count, 3), dtype=xp.float64)
    for mode in range(3):
        pure[:, mode] = xp.where(values["pure_mode"] == mode, 1.0, 0.0)
    pair = xp.zeros((count, 3), dtype=xp.float64)
    fraction = values["pair_mixing"]
    # pair 0=(scalar,vector), 1=(scalar,tensor), 2=(vector,tensor)
    pair[:, 0] = xp.where(values["pair_mode"] == 0, fraction, xp.where(values["pair_mode"] == 1, fraction, 0.0))
    pair[:, 1] = xp.where(values["pair_mode"] == 0, 1.0 - fraction, xp.where(values["pair_mode"] == 2, fraction, 0.0))
    pair[:, 2] = xp.where(values["pair_mode"] == 1, 1.0 - fraction, xp.where(values["pair_mode"] == 2, 1.0 - fraction, 0.0))
    triple = values["triple_simplex"]
    return xp.where(
        (values["niche"] == 0)[:, None],
        pure,
        xp.where((values["niche"] == 1)[:, None], pair, triple),
    )


def _basis(u: Any, xp: Any) -> Any:
    u = xp.maximum(u, 0.0)
    return xp.clip(1.0 - (1.0 + u) * xp.exp(-u), 0.0, 1.0)


def _candidate_log_mu(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    values = _candidate_values(config, arrays, begin, end, xp)
    weights = _mode_weights(values, xp)
    radii = xp.asarray(
        np.asarray([[row["reff_kpc"], row["rein_kpc"]] for row in rows], dtype=np.float64)
    )[None, :, :]
    lambdas = xp.stack(
        [
            values["lambda_scalar"],
            values["lambda_scalar"] * values["range_vector"],
            values["lambda_scalar"] * values["range_tensor"],
        ],
        axis=1,
    )
    mode_basis = _basis(radii[:, :, :, None] / lambdas[:, None, None, :], xp)
    projectors = xp.asarray(
        [row["lensing_to_matter"] for row in config["physics"]["helicity_projectors"]]
    )
    matter = xp.sum(mode_basis * weights[:, None, None, :], axis=3)
    light = xp.sum(mode_basis * weights[:, None, None, :] * projectors[None, None, None, :], axis=3)
    pairs = ((0, 1), (0, 2), (1, 2))
    cross = xp.zeros_like(matter)
    cross_light = xp.zeros_like(light)
    for pair_index, (first, second) in enumerate(pairs):
        active = values["interaction_pair"] == pair_index
        product = 2.0 * xp.sqrt(
            xp.maximum(
                weights[:, first, None, None]
                * weights[:, second, None, None]
                * mode_basis[:, :, :, first]
                * mode_basis[:, :, :, second],
                0.0,
            )
        )
        cross = xp.where(active[:, None, None], product, cross)
        p_cross = 0.5 * (float(projectors[first]) + float(projectors[second]))
        cross_light = xp.where(active[:, None, None], p_cross * product, cross_light)
    interaction = (
        (values["niche"] == 3)
        * values["interaction_fraction"]
        * values["interaction_phase"]
    )
    matter = matter + interaction[:, None, None] * cross
    light = light + interaction[:, None, None] * cross_light
    coefficient = values["polarity"] * values["amplitude"]
    log_matter = coefficient[:, None, None] * matter
    log_light = coefficient[:, None, None] * light
    return xp.stack([log_matter[:, :, 0], log_light[:, :, 1]], axis=2)


def _exact_signature_classes(config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> int:
    signatures: set[tuple[int, ...]] = set()
    for index in range(len(arrays["niche"])):
        niche = int(arrays["niche"][index])
        common = (
            niche,
            int(arrays["polarity"][index]),
            int(arrays["amplitude"][index]),
        )
        if niche == 0:
            mode = int(arrays["pure_mode"][index])
            relevant_range = (
                -1
                if mode == 0
                else int(arrays["range_vector"][index])
                if mode == 1
                else int(arrays["range_tensor"][index])
            )
            signature = common + (int(arrays["lambda_scalar"][index]), mode, relevant_range)
        elif niche == 1:
            signature = common + (
                int(arrays["lambda_scalar"][index]),
                int(arrays["range_vector"][index]),
                int(arrays["range_tensor"][index]),
                int(arrays["pair_mode"][index]),
                int(arrays["pair_mixing"][index]),
            )
        elif niche == 2:
            signature = common + (
                int(arrays["lambda_scalar"][index]),
                int(arrays["range_vector"][index]),
                int(arrays["range_tensor"][index]),
                int(arrays["triple_simplex"][index]),
            )
        else:
            signature = common + (
                int(arrays["lambda_scalar"][index]),
                int(arrays["range_vector"][index]),
                int(arrays["range_tensor"][index]),
                int(arrays["triple_simplex"][index]),
                int(arrays["interaction_pair"][index]),
                int(arrays["interaction_fraction"][index]),
                int(arrays["interaction_phase"][index]),
            )
        signatures.add(signature)
    return len(signatures)


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    arrays = generate_candidates(config)
    niches = Counter(int(value) for value in arrays["niche"])
    polarities = Counter(int(value) for value in arrays["polarity"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item22-polarization-candidates-1.0",
            "response_values_read": 0,
            "post_response_candidate_cells": 0,
            "raw_candidate_cells": len(arrays["niche"]),
            "niche_counts": {
                str(config["candidate_generator"]["niches"][key]["id"]): value
                for key, value in sorted(niches.items())
            },
            "polarity_counts": {
                str(config["candidate_generator"]["polarities"][key]): value
                for key, value in sorted(polarities.items())
            },
            "all_three_range_candidates_are_distinct": True,
            "exact_parameter_signature_classes": _exact_signature_classes(config, arrays),
            "candidate_array_sha256": _candidate_digest(arrays),
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "historical_novelty_claimed": False,
        }
    )


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["identity_catalog"].parent.mkdir(parents=True, exist_ok=True)
    identity_body, identity_headers = _download(str(config["sources"]["identity_query"]))
    paths["identity_catalog"].write_bytes(identity_body)
    columns = tuple(config["sources"]["identity_columns"])
    identities = _parse_vizier_tsv(identity_body, columns)
    if len(identities) != int(config["sources"]["expected_identity_rows"]):
        raise GravityItem22Error(f"identity catalog changed: {len(identities)}")
    prior = _prior_identities(root, config)
    preview = {_normal_identity(value) for value in config["sources"]["preview_exposed_identities"]}
    prior_excluded: list[str] = []
    preview_excluded: list[str] = []
    missing_coordinates: list[str] = []
    query_rows: list[dict[str, str]] = []
    for row in identities:
        normalized = _normal_identity(row["Name"])
        if normalized in prior:
            prior_excluded.append(str(row["Name"]).strip())
        elif normalized in preview:
            preview_excluded.append(str(row["Name"]).strip())
        else:
            try:
                float(row["_RA"])
                float(row["_DE"])
                query_rows.append(row)
            except ValueError:
                missing_coordinates.append(str(row["Name"]).strip())
    expected = config["sources"]
    checks = (
        (len(prior_excluded), int(expected["expected_prior_identity_exclusions"]), "prior identities"),
        (len(preview_excluded), int(expected["expected_preview_exclusions_after_prior"]), "preview identities"),
        (len(missing_coordinates), int(expected["expected_missing_coordinate_exclusions_after_prior_and_preview"]), "missing coordinates"),
        (len(query_rows), int(expected["expected_legacy_queries"]), "Legacy queries"),
    )
    for actual, wanted, label in checks:
        if actual != wanted:
            raise GravityItem22Error(f"{label} changed: {actual} != {wanted}")
    extracted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for index, identity in enumerate(query_rows):
        ra, dec = float(identity["_RA"]), float(identity["_DE"])
        url = _legacy_url(config, ra, dec)
        body, headers = _download(url)
        tractor, reasons = _nearest_legacy_row(body, ra, dec, config)
        receipts.append(
            {
                "name": str(identity["Name"]).strip(),
                "url": url,
                "bytes": len(body),
                "sha256": _sha256_bytes(body),
                "last_modified": headers.get("last-modified"),
                "etag": headers.get("etag"),
                "quality_pass": tractor is not None and not reasons,
                "quality_failures": reasons,
            }
        )
        if tractor is not None and not reasons:
            extracted.append(_derived_predictor(identity, tractor, config))
        else:
            failures.append(
                {"name": str(identity["Name"]).strip(), "quality_failures": reasons}
            )
        if (index + 1) % 10 == 0:
            print(f"Item 22 predictor queries {index + 1}/{len(query_rows)}", flush=True)
    if len(extracted) != int(expected["expected_predictor_quality_eligible"]):
        raise GravityItem22Error(f"predictor eligibility changed: {len(extracted)}")
    predictor_columns = list(extracted[0])
    _write_tsv(paths["legacy_predictors"], extracted, predictor_columns)
    sample_manifest = _build_sample(extracted, config)
    candidate_manifest = _candidate_manifest(config)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item22-predictor-source-1.0",
            "response_values_read": 0,
            "selection_used_response_values": False,
            "identity_source": {
                "url": config["sources"]["identity_query"],
                "bytes": len(identity_body),
                "sha256": _sha256_bytes(identity_body),
                "last_modified": identity_headers.get("last-modified"),
                "etag": identity_headers.get("etag"),
                "approved_columns": list(columns),
            },
            "legacy_source": {
                "release": config["sources"]["legacy_release"],
                "documentation": config["sources"]["legacy_catalog_documentation"],
                "per_object_receipts": receipts,
            },
            "counts": {
                "identity_rows": len(identities),
                "prior_identity_exclusions": len(prior_excluded),
                "preview_exclusions": len(preview_excluded),
                "missing_coordinate_exclusions": len(missing_coordinates),
                "legacy_queries": len(query_rows),
                "predictor_quality_failures": len(failures),
                "predictor_quality_eligible": len(extracted),
            },
            "prior_identity_exclusions": sorted(prior_excluded),
            "preview_exclusions": sorted(preview_excluded),
            "missing_coordinate_exclusions": sorted(missing_coordinates),
            "predictor_quality_failures": sorted(failures, key=lambda row: row["name"]),
            "predictor_file": {
                "path": paths["legacy_predictors"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["legacy_predictors"]),
            },
            "claim_boundary": config["scope"]["claim_ceiling"],
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], candidate_manifest)
    return paths


def _load_prepared(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictor = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    for value, label in (
        (predictor, "predictor manifest"),
        (sample, "sample manifest"),
        (candidates, "candidate manifest"),
    ):
        _verify_content_hash(value, label)
        if int(value.get("response_values_read", 0)) != 0:
            raise GravityItem22Error(f"{label} contains response values")
    if _sha256_file(paths["legacy_predictors"]) != predictor["predictor_file"]["sha256"]:
        raise GravityItem22Error("predictor TSV changed")
    arrays = generate_candidates(config)
    if _candidate_digest(arrays) != candidates["candidate_array_sha256"]:
        raise GravityItem22Error("candidate array changed")
    return predictor, sample, candidates


def _response_url(config: Mapping[str, Any], name: str) -> str:
    params = {
        "-source": "J/MNRAS/498/6013/tablea1",
        "-out": ",".join(config["sources"]["response_columns"]),
        "Name": name,
        "-out.max": "2",
    }
    return str(config["sources"]["response_query_base"]) + "?" + urllib.parse.urlencode(params)


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    _, sample, candidates = _load_prepared(root, config)
    if int(candidates["post_response_candidate_cells"]) != 0:
        raise GravityItem22Error("post-response candidates detected")
    paths = _source_paths(root, config)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    columns = tuple(config["sources"]["response_columns"])
    responses: list[dict[str, str]] = []
    receipts: list[dict[str, Any]] = []
    for row in exploration:
        name = str(row["name"])
        url = _response_url(config, name)
        body, headers = _download(url)
        parsed = _parse_vizier_tsv(body, columns)
        exact = [value for value in parsed if str(value["Name"]).strip() == name]
        if len(exact) != 1:
            raise GravityItem22Error(f"response query did not return one exact row: {name}")
        responses.append(exact[0])
        receipts.append(
            {
                "name": name,
                "role": "exploration",
                "url": url,
                "bytes": len(body),
                "sha256": _sha256_bytes(body),
                "last_modified": headers.get("last-modified"),
                "etag": headers.get("etag"),
            }
        )
    _write_tsv(paths["exploration_responses"], responses, columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item22-response-source-1.0",
            "confirmation_response_values_read": 0,
            "exploration_response_rows": len(responses),
            "post_response_candidate_cells": 0,
            "columns": list(columns),
            "forbidden_columns_read": [],
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
            "per_object_source_receipts": receipts,
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["exploration_responses"]


def _load_response_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    _, sample, _ = _load_prepared(root, config)
    manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(manifest, "response manifest")
    if int(manifest["confirmation_response_values_read"]) != 0:
        raise GravityItem22Error("confirmation response was opened")
    if _sha256_file(paths["exploration_responses"]) != manifest["response_file"]["sha256"]:
        raise GravityItem22Error("response TSV changed")
    response = {row["Name"]: row for row in _read_tsv(paths["exploration_responses"])}
    rows: list[dict[str, Any]] = []
    for predictor in sample["objects"]:
        if predictor["role"] != "exploration":
            continue
        name = str(predictor["name"])
        observed = response.get(name)
        if observed is None:
            continue
        try:
            sigma = float(observed["sigma"])
            upper = float(observed["E_sigma"])
            lower = float(observed["e_sigma"])
            theta = float(observed["thetaE"])
            theta_error = float(observed["e_thetaE"]) if observed["e_thetaE"].strip() else 0.0
            z_lens = float(predictor["z_lens"])
            z_source = float(predictor["z_source"])
            reff = float(predictor["reff_kpc"])
            luminosity = float(predictor["z_luminosity_Lsun"])
        except (TypeError, ValueError):
            continue
        if not (
            sigma > 0
            and upper > 0
            and lower > 0
            and theta > 0
            and z_source > z_lens > 0
            and reff > 0
            and luminosity > 0
        ):
            continue
        d_lens, d_source, d_lens_source = _angular_diameter_distances(z_lens, z_source, config)
        rein = theta * float(config["physics"]["constants"]["arcsec_to_radian"]) * d_lens
        re_over_a = float(config["physics"]["hernquist_re_over_a"])
        projected_fraction = float(hernquist_projected_mass_fraction(rein / (reff / re_over_a)))
        gravitational = float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
        c = float(config["physics"]["constants"]["c_km_s"])
        sigma_critical = (c**2 / (4.0 * math.pi * gravitational)) * (
            d_source / (d_lens * d_lens_source)
        )
        required_lens_mass = math.pi * rein**2 * sigma_critical
        virial = float(config["physics"]["dynamical_virial_coefficient"])
        y_dyn = math.log(virial * reff * sigma**2 / (gravitational * luminosity))
        y_lens = math.log(required_lens_mass / (luminosity * projected_fraction))
        rows.append(
            {
                **{key: predictor[key] for key in predictor},
                "fold": int(predictor["fold"]),
                "sigma_km_s": sigma,
                "sigma_error_km_s": 0.5 * (upper + lower),
                "theta_ein_arcsec": theta,
                "theta_ein_error_arcsec": theta_error,
                "rein_kpc": rein,
                "projected_fraction_at_rein": projected_fraction,
                "y_dyn": y_dyn,
                "y_lens": y_lens,
            }
        )
    rows.sort(key=lambda row: str(row["name"]))
    return rows, manifest


def _backend() -> tuple[Any, str, str]:
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return cp, "gpu_cupy", str(name)
    except (ImportError, RuntimeError, OSError) as error:
        raise GravityItem22Error(f"Item 22 requires the frozen CUDA lane: {error}") from error


def _to_numpy(value: Any, xp: Any) -> np.ndarray:
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def _build_log_mu_matrix(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], xp: Any
) -> Any:
    batch = int(config["evaluation"]["candidate_batch_size"])
    pieces = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(_candidate_log_mu(config, arrays, rows, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _fit_offset(values: np.ndarray, bounds: tuple[float, float]) -> tuple[float, float]:
    raw = float(np.mean(values))
    return raw, float(np.clip(raw, math.log(bounds[0]), math.log(bounds[1])))


def _screen_log_mu(
    log_mu: Any, y: np.ndarray, folds: np.ndarray, config: Mapping[str, Any], xp: Any
) -> dict[str, Any]:
    bounds = tuple(float(value) for value in config["physics"]["shared_mass_to_light_bounds_Msun_per_Lzsun"])
    y_device = xp.asarray(y)
    prediction = np.empty_like(y)
    selected: list[int] = []
    offsets: list[float] = []
    raw_offsets: list[float] = []
    training_mse: list[float] = []
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        residual = y_device[None, train, :] - log_mu[:, train, :]
        raw = xp.mean(residual, axis=(1, 2))
        fitted = xp.clip(raw, math.log(bounds[0]), math.log(bounds[1]))
        mse = xp.mean((residual - fitted[:, None, None]) ** 2, axis=(1, 2))
        index = int(_to_numpy(xp.argmin(mse), xp))
        selected.append(index)
        raw_offsets.append(float(_to_numpy(raw[index], xp)))
        offsets.append(float(_to_numpy(fitted[index], xp)))
        training_mse.append(float(_to_numpy(mse[index], xp)))
        prediction[held] = _to_numpy(log_mu[index, held, :], xp) + offsets[-1]
    return {
        "prediction": prediction,
        "selected_indices": selected,
        "log_mass_to_light_offsets": offsets,
        "raw_log_mass_to_light_offsets": raw_offsets,
        "training_mse": training_mse,
    }


def _feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    surveys = ("BELLS", "SLACS", "SL2S", "CASTLES", "DES", "LSD")
    values = []
    for row in rows:
        base = [
            math.log(float(row["z_luminosity_Lsun"])),
            math.log(float(row["reff_kpc"])),
            float(row["z_lens"]),
            float(row["g_minus_r"]),
            float(row["r_minus_z"]),
            float(row["axis_ratio"]),
            float(row["max_fracflux_grz"]),
        ]
        base.extend(float(str(row["survey"]) == survey) for survey in surveys)
        values.append(base)
    return np.asarray(values, dtype=np.float64)


def _baseline_predictions(
    y: np.ndarray, folds: np.ndarray, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    bounds = tuple(float(value) for value in config["physics"]["shared_mass_to_light_bounds_Msun_per_Lzsun"])
    shared = np.empty_like(y)
    separate = np.empty_like(y)
    flexible = np.empty_like(y)
    feature = _feature_matrix(rows)
    alpha = float(config["evaluation"]["ridge_alpha"])
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        _, offset = _fit_offset(y[train].reshape(-1), bounds)
        shared[held] = offset
        for channel in range(2):
            _, channel_offset = _fit_offset(y[train, channel], bounds)
            separate[held, channel] = channel_offset
        mean = feature[train].mean(axis=0)
        scale = feature[train].std(axis=0)
        scale[scale == 0] = 1.0
        train_design = np.column_stack([np.ones(len(train)), (feature[train] - mean) / scale])
        held_design = np.column_stack([np.ones(len(held)), (feature[held] - mean) / scale])
        penalty = np.diag([0.0] + [alpha] * feature.shape[1])
        for channel in range(2):
            coefficient = np.linalg.solve(
                train_design.T @ train_design + penalty,
                train_design.T @ y[train, channel],
            )
            flexible[held, channel] = held_design @ coefficient
    return {"shared_GR": shared, "separate_calibration": separate, "flexible_nuisance": flexible}


def _mse(y: np.ndarray, prediction: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is not None:
        y, prediction = y[indices], prediction[indices]
    return float(np.mean((y - prediction) ** 2))


def _improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference if reference > 0 else float("-inf")


def _selected_cell(index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    g = config["candidate_generator"]
    niche = int(arrays["niche"][index])
    values = _candidate_values(config, arrays, index, index + 1, np)
    weights = _mode_weights(values, np)[0]
    return {
        "candidate_index": index,
        "niche": g["niches"][niche],
        "polarity": float(values["polarity"][0]),
        "amplitude": float(values["amplitude"][0]),
        "weights_scalar_vector_tensor": [float(value) for value in weights],
        "lambda_scalar_kpc": float(values["lambda_scalar"][0]),
        "lambda_vector_kpc": float(values["lambda_scalar"][0] * values["range_vector"][0]),
        "lambda_tensor_kpc": float(values["lambda_scalar"][0] * values["range_tensor"][0]),
        "interaction_pair": int(values["interaction_pair"][0]) if niche == 3 else None,
        "interaction_fraction": float(values["interaction_fraction"][0]) if niche == 3 else 0.0,
        "interaction_phase": float(values["interaction_phase"][0]) if niche == 3 else 0.0,
    }


def _local_limit_max(config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], xp: Any) -> float:
    radius = float(config["physics"]["constants"]["au_to_kpc"])
    rows = [{"reff_kpc": radius, "rein_kpc": radius}]
    maximum = 0.0
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        values = _candidate_log_mu(config, arrays, rows, begin, end, xp)
        maximum = max(maximum, float(_to_numpy(xp.max(xp.abs(xp.expm1(values))), xp)))
    return maximum


def _synthetic_controls(
    log_mu: Any,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    xp: Any,
) -> dict[str, Any]:
    niche3 = np.where(arrays["niche"] == 3)[0]
    injection_index = int(niche3[len(niche3) // 2])
    injection = _to_numpy(log_mu[injection_index], xp)
    y_injection = math.log(2.0) + injection
    injected = _screen_log_mu(log_mu, y_injection, folds, config, xp)
    injected_niches = [int(arrays["niche"][index]) for index in injected["selected_indices"]]
    baseline = _baseline_predictions(y_injection, folds, rows, config)["flexible_nuisance"]
    injected_mse = _mse(y_injection, injected["prediction"])
    baseline_mse = _mse(y_injection, baseline)
    y_gr = np.full((len(rows), 2), math.log(2.0), dtype=np.float64)
    gr = _screen_log_mu(log_mu, y_gr, folds, config, xp)
    gr_baseline = _baseline_predictions(y_gr, folds, rows, config)["shared_GR"]
    gr_candidate_mse = _mse(y_gr, gr["prediction"])
    gr_baseline_mse = _mse(y_gr, gr_baseline)
    return {
        "injection_candidate_index": injection_index,
        "injection_selected_niches": injected_niches,
        "injection_exact_niche_recovered_all_folds": all(value == 3 for value in injected_niches),
        "injection_candidate_mse": injected_mse,
        "injection_flexible_mse": baseline_mse,
        "injection_improves_over_flexible": injected_mse < baseline_mse,
        "GR_candidate_mse": gr_candidate_mse,
        "GR_baseline_mse": gr_baseline_mse,
        "GR_control_prefers_nonzero_polarization": gr_candidate_mse < gr_baseline_mse - 1e-18,
    }


def _weighted_mse(
    y: np.ndarray, prediction: np.ndarray, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> float:
    lum_error = math.log(10.0) * float(config["evaluation"]["stellar_luminosity_systematic_dex"])
    fallback = float(config["evaluation"]["lens_radius_fractional_uncertainty_when_missing"])
    errors = []
    for row in rows:
        sigma_fraction = float(row["sigma_error_km_s"]) / float(row["sigma_km_s"])
        theta_error = float(row["theta_ein_error_arcsec"])
        theta_fraction = theta_error / float(row["theta_ein_arcsec"]) if theta_error > 0 else fallback
        errors.append([math.hypot(2.0 * sigma_fraction, lum_error), math.hypot(2.0 * theta_fraction, lum_error)])
    weights = 1.0 / np.asarray(errors, dtype=np.float64) ** 2
    return float(np.sum(weights * (y - prediction) ** 2) / np.sum(weights))


def _evaluate(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    record_compute: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum = int(config["sample"]["minimum_complete_exploration_objects"])
    if len(rows) < minimum:
        raise GravityItem22Error(f"too few complete exploration objects: {len(rows)} < {minimum}")
    arrays = generate_candidates(config)
    xp, backend, device = _backend()
    y = np.asarray([[row["y_dyn"], row["y_lens"]] for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if set(folds.tolist()) != set(range(int(config["sample"]["outer_folds"]))):
        raise GravityItem22Error("exploration folds are incomplete")
    xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    log_mu = _build_log_mu_matrix(config, arrays, rows, xp)
    xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    crosscheck = int(config["evaluation"]["cpu_crosscheck_candidates"])
    cpu = _candidate_log_mu(config, arrays, rows, 0, crosscheck, np)
    gpu = _to_numpy(log_mu[:crosscheck], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu - gpu)))
    local_limit = _local_limit_max(config, arrays, xp)
    controls = _synthetic_controls(log_mu, folds, rows, config, arrays, xp)
    start = time.perf_counter()
    selected = _screen_log_mu(log_mu, y, folds, config, xp)
    baselines = _baseline_predictions(y, folds, rows, config)
    candidate_mse = _mse(y, selected["prediction"])
    baseline_mse = {key: _mse(y, value) for key, value in baselines.items()}
    observed_statistic = _improvement(baseline_mse["flexible_nuisance"], candidate_mse)
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    null_statistics: list[float] = []
    trials = int(config["evaluation"]["permutation_trials"])
    for trial in range(trials):
        permuted_y = y[random.permutation(len(rows))]
        permuted = _screen_log_mu(log_mu, permuted_y, folds, config, xp)
        flexible = _baseline_predictions(permuted_y, folds, rows, config)["flexible_nuisance"]
        null_statistics.append(_improvement(_mse(permuted_y, flexible), _mse(permuted_y, permuted["prediction"])))
        if record_compute and (trial + 1) % 10 == 0:
            print(f"Item 22 selection-aware null {trial + 1}/{trials}", flush=True)
    xp.cuda.Stream.null.synchronize()
    screen_seconds = time.perf_counter() - start
    raw_p = (1 + sum(value >= observed_statistic for value in null_statistics)) / (trials + 1)
    guarded_p = 1.0 if observed_statistic <= 0.0 else raw_p
    cells = [_selected_cell(index, config, arrays) for index in selected["selected_indices"]]
    niche_counts = Counter(str(cell["niche"]["id"]) for cell in cells)
    novel_counts = {
        key: value
        for key, value in niche_counts.items()
        if bool(next(n for n in config["candidate_generator"]["niches"] if n["id"] == key)["novel_relative_to_item16"])
    }
    stable_novel = max(novel_counts.values(), default=0)
    channel_metrics: dict[str, Any] = {}
    for channel, label in enumerate(("stellar_dynamics", "Einstein_radius_lensing")):
        value = {"candidate_mse": float(np.mean((y[:, channel] - selected["prediction"][:, channel]) ** 2))}
        for key, prediction in baselines.items():
            mse = float(np.mean((y[:, channel] - prediction[:, channel]) ** 2))
            value[f"{key}_mse"] = mse
            value[f"improvement_vs_{key}"] = _improvement(mse, value["candidate_mse"])
        channel_metrics[label] = value
    strata: dict[str, Any] = {}
    for key, label in (("z_luminosity_Lsun", "luminosity"), ("reff_kpc", "size"), ("z_lens", "redshift")):
        values = np.asarray([float(row[key]) for row in rows])
        median = float(np.median(values))
        for side, indices in (("low", np.where(values <= median)[0]), ("high", np.where(values > median)[0])):
            cand = _mse(y, selected["prediction"], indices)
            shared = _mse(y, baselines["shared_GR"], indices)
            flexible = _mse(y, baselines["flexible_nuisance"], indices)
            strata[f"{label}_{side}"] = {
                "objects": len(indices),
                "candidate_mse": cand,
                "shared_GR_mse": shared,
                "flexible_nuisance_mse": flexible,
                "improvement_vs_shared_GR": _improvement(shared, cand),
                "improvement_vs_flexible_nuisance": _improvement(flexible, cand),
            }
    survey_metrics: dict[str, Any] = {}
    for survey in sorted({str(row["survey"]) for row in rows}):
        indices = np.where(np.asarray([str(row["survey"]) == survey for row in rows]))[0]
        survey_metrics[survey] = {
            "objects": len(indices),
            "improvement_vs_shared_GR": _improvement(
                _mse(y, baselines["shared_GR"], indices), _mse(y, selected["prediction"], indices)
            ),
            "improvement_vs_flexible_nuisance": _improvement(
                _mse(y, baselines["flexible_nuisance"], indices), _mse(y, selected["prediction"], indices)
            ),
        }
    bounds = tuple(float(value) for value in config["physics"]["shared_mass_to_light_bounds_Msun_per_Lzsun"])
    raw_scales = [math.exp(value) for value in selected["raw_log_mass_to_light_offsets"]]
    mass_to_light_in_bounds = all(bounds[0] <= value <= bounds[1] for value in raw_scales)
    g = config["gates"]
    gates = {
        "minimum_complete_exploration_objects": len(rows) >= minimum,
        "confirmation_values_read_zero": True,
        "post_response_candidate_cells_zero": int(config["candidate_generator"]["post_response_cells"]) == 0,
        "local_classical_limit": local_limit <= float(g["max_local_fractional_deviation_at_1AU"]),
        "positive_matter_and_light_response": True,
        "synthetic_injection_recovers_interaction_niche": bool(controls["injection_exact_niche_recovered_all_folds"]) and bool(controls["injection_improves_over_flexible"]),
        "known_GR_control": not bool(controls["GR_control_prefers_nonzero_polarization"]),
        "joint_improvement_vs_shared_GR": _improvement(baseline_mse["shared_GR"], candidate_mse) >= float(g["minimum_joint_mse_improvement_vs_shared_GR"]),
        "joint_improvement_vs_separate_calibration": _improvement(baseline_mse["separate_calibration"], candidate_mse) >= float(g["minimum_joint_mse_improvement_vs_separate_calibration"]),
        "joint_improvement_vs_flexible_nuisance": observed_statistic > float(g["minimum_joint_mse_improvement_vs_flexible_nuisance"]),
        "both_channels_improve_vs_shared_GR": all(value["improvement_vs_shared_GR"] > float(g["minimum_each_channel_improvement_vs_shared_GR"]) for value in channel_metrics.values()),
        "all_broad_halves_improve_vs_shared_GR": all(value["improvement_vs_shared_GR"] > float(g["minimum_each_broad_half_improvement_vs_shared_GR"]) for value in strata.values()),
        "selection_aware_permutation": guarded_p <= float(g["maximum_selection_aware_permutation_p"]),
        "stable_novel_niche": stable_novel >= int(g["minimum_same_novel_niche_folds"]),
        "shared_mass_to_light_in_bounds": mass_to_light_in_bounds,
    }
    phenomenon_gates = {
        "beats_flexible_by_five_percent": observed_statistic >= float(g["phenomenon_minimum_improvement_vs_flexible"]),
        "selection_aware_significance": guarded_p <= float(g["maximum_selection_aware_permutation_p"]),
        "stable_novel_niche": stable_novel >= int(g["minimum_same_novel_niche_folds"]),
        "at_least_one_channel_beats_flexible": any(value["improvement_vs_flexible_nuisance"] > 0 for value in channel_metrics.values()),
        "pipeline_controls": bool(controls["injection_exact_niche_recovered_all_folds"]) and not bool(controls["GR_control_prefers_nonzero_polarization"]),
    }
    universal_advance = all(gates.values())
    phenomenon_lead = all(phenomenon_gates.values())
    counterexamples = [
        str(row["name"])
        for index, row in enumerate(rows)
        if float(np.mean((y[index] - selected["prediction"][index]) ** 2))
        > float(np.mean((y[index] - baselines["flexible_nuisance"][index]) ** 2))
    ]
    score_evaluations = len(arrays["niche"]) * 2 * sum(
        int(np.count_nonzero(folds != fold)) for fold in range(int(config["sample"]["outer_folds"]))
    )
    compute = {
        "schema_version": "invariant-gravity-item22-compute-1.0",
        "backend": backend,
        "device": device,
        "numpy_version": np.__version__,
        "cupy_version": getattr(xp, "__version__", None),
        "candidate_cells": len(arrays["niche"]),
        "objects": len(rows),
        "channels": 2,
        "candidate_observable_matrix_values": int(np.prod(log_mu.shape)),
        "candidate_training_residual_evaluations_observed": score_evaluations,
        "candidate_training_residual_evaluations_with_nulls": score_evaluations * (trials + 1),
        "matrix_build_seconds_observed": matrix_seconds,
        "screen_and_null_seconds_observed": screen_seconds,
        "cpu_gpu_max_absolute_log_mu_difference": cpu_gpu_max,
    }
    scientific = {
        "decision": "PASS_ITEM22_POLARIZATION_EXPLORATION" if universal_advance else "REJECT_ITEM22_POLARIZATION_EXPLORATION",
        "track_decisions": {
            "universal_gravity": "ADVANCE" if universal_advance else "DO_NOT_ADVANCE",
            "phenomenon_publication": "REPLICATION_LEAD" if phenomenon_lead else "NO_EMPIRICAL_LEAD",
            "paper_claim_now": False,
        },
        "counts": {
            "valid_exploration_objects": len(rows),
            "candidate_cells": len(arrays["niche"]),
            "post_response_candidate_cells": 0,
            "permutation_trials": trials,
            "passed_universal_gates": sum(bool(value) for value in gates.values()),
            "total_universal_gates": len(gates),
            "individual_counterexamples_vs_flexible": len(counterexamples),
        },
        "primary_metrics": {
            "candidate_mse": candidate_mse,
            **{f"{key}_mse": value for key, value in baseline_mse.items()},
            "improvement_vs_shared_GR": _improvement(baseline_mse["shared_GR"], candidate_mse),
            "improvement_vs_separate_calibration": _improvement(baseline_mse["separate_calibration"], candidate_mse),
            "improvement_vs_flexible_nuisance": observed_statistic,
            "selection_aware_raw_permutation_p": raw_p,
            "selection_aware_guarded_permutation_p": guarded_p,
        },
        "weighted_robustness": {
            "candidate_mse": _weighted_mse(y, selected["prediction"], rows, config),
            **{f"{key}_mse": _weighted_mse(y, value, rows, config) for key, value in baselines.items()},
        },
        "channel_metrics": channel_metrics,
        "stratum_metrics": strata,
        "survey_metrics": survey_metrics,
        "outer_selections": [
            {
                "fold": fold,
                "cell": cells[fold],
                "training_mse": selected["training_mse"][fold],
                "mass_to_light": math.exp(selected["log_mass_to_light_offsets"][fold]),
                "unclipped_mass_to_light": raw_scales[fold],
                "heldout_objects": [str(rows[index]["name"]) for index in np.where(folds == fold)[0]],
            }
            for fold in range(int(config["sample"]["outer_folds"]))
        ],
        "selection_stability": {
            "niche_counts": dict(sorted(niche_counts.items())),
            "maximum_same_novel_niche_folds": stable_novel,
            "exact_candidate_indices": selected["selected_indices"],
        },
        "null_distribution": {
            "statistic": "OOF improvement versus fixed flexible photometric nuisance model",
            "observed": observed_statistic,
            "minimum": min(null_statistics),
            "median": float(np.median(null_statistics)),
            "maximum": max(null_statistics),
            "sha256": _sha256_bytes(np.asarray(null_statistics, dtype="<f8").tobytes()),
        },
        "controls": {
            **controls,
            "maximum_fractional_deviation_at_1AU": local_limit,
            "cpu_gpu_max_absolute_log_mu_difference": cpu_gpu_max,
        },
        "universal_gates": gates,
        "phenomenon_publication_gates": phenomenon_gates,
        "counterexample_names_vs_flexible": counterexamples,
    }
    del log_mu
    xp.get_default_memory_pool().free_all_blocks()
    return scientific, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    response_manifest: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    predictor, sample, candidates = _load_prepared(root, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item22-polarization-superposition-receipt-1.0",
            "item": 22,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "discovery_policy": config["discovery_policy"],
            "mathematical_definition": {
                "helicity_projectors": config["physics"]["helicity_projectors"],
                "carrier_basis": config["physics"]["carrier_basis"],
                "matter_response": config["physics"]["matter_response"],
                "lensing_response": config["physics"]["lensing_response"],
                "dynamics_mass_balance": "log[5 Re sigma^2/(G Lz)] = log(M/Lz)+log(mu_D(Re))",
                "lensing_mass_balance": "log[pi RE^2 SigmaCrit/(Lz f_H(RE))] = log(M/Lz)+log(mu_L(RE))",
            },
            "provenance_and_creativity_labels": config["candidate_generator"]["niches"],
            "equivalence_audit": {
                "item16_controls_retained": [
                    "single_helicity_projector_control",
                    "two_helicity_two_range_linear_control",
                ],
                "non_equivalent_search_regions": [
                    "three_helicity_three_range_superposition",
                    "three_helicity_vertex_interference",
                ],
                "historical_novelty_claimed": False,
                "boundaries": config["candidate_generator"]["equivalence_boundaries"],
            },
            "claim_boundary": config["scope"]["claim_ceiling"],
            "source_bindings": {
                "predictor_manifest_sha256": predictor["content_sha256"],
                "sample_manifest_sha256": sample["content_sha256"],
                "candidate_manifest_sha256": candidates["content_sha256"],
                "response_manifest_sha256": response_manifest["content_sha256"],
                "response_file_sha256": response_manifest["response_file"]["sha256"],
                "observable_lineage": config["sources"]["observable_lineage"],
            },
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "stable_goal_sha256": config["stable_goal_sha256"],
                "confirmation_opened": False,
                "confirmation_response_values_read": 0,
                "post_response_formula_generation": False,
            },
            "baselines": {
                "shared_GR": config["evaluation"]["baseline_shared_GR"],
                "separate_calibration": config["evaluation"]["baseline_separate_calibration"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
            },
            "scientific_result": scientific,
            "compute_and_api_cost": {**compute, "paid_model_calls": 0, "paid_api_spend_usd": 0.0},
            "counterexamples_and_limitations": [
                "Legacy DR9 Tractor light can contain residual lens-arc or source contamination; fracflux and mask cuts reduce but cannot eliminate it.",
                "A z-band luminosity proxy with one training-fold mass-to-light scale is weaker than object-level stellar-population mass inference.",
                "The spherical Hernquist and virial mappings are not a resolved Jeans analysis and can create dynamics systematics.",
                "Published SIS/SIE Einstein radii are model-derived evaluation radii, not the roadmap's final direct image likelihood.",
                "The exponential response and interaction vertex are weak-field ansatzes, not a covariant, conserved, causal, or ghost-free theory.",
                "No sealed confirmation response is opened; a positive exploration result is only a replication lead.",
            ],
            "exact_next_action": "If neither track advances, preserve the tested equivalence and counterexample region and advance to Item 23 bimetric gravity on fresh data. If the phenomenon track advances, preregister an unchanged cross-source replication without pausing the numbered gravity track.",
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
                "valid_object_names": [str(row["name"]) for row in rows],
            },
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_response_rows(root, config)
    scientific, compute_raw = _evaluate(root, config, rows, record_compute=True)
    paths = _source_paths(root, config)
    compute = _content_hashed(compute_raw)
    _write_json(paths["compute_manifest"], compute)
    receipt = _build_receipt(root, config, rows, response_manifest, scientific, compute)
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, receipt)
    return result_path


def validate_result(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    for key in (
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "response_source_manifest",
        "compute_manifest",
    ):
        value = _read_json(paths[key])
        _verify_content_hash(value, key)
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem22Error("result opened confirmation data")
    if bool(result["equivalence_audit"]["historical_novelty_claimed"]):
        raise GravityItem22Error("result made an unauthorized historical novelty claim")
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-predictors")
    sub.add_parser("acquire-responses")
    sub.add_parser("run")
    sub.add_parser("validate")
    sub.add_parser("show-candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "prepare-predictors":
        paths = prepare_predictors(root)
        print(paths["sample_manifest"].as_posix())
    elif args.command == "acquire-responses":
        print(acquire_responses(root).as_posix())
    elif args.command == "run":
        print(run_experiment(root).as_posix())
    elif args.command == "validate":
        print(validate_result(root).as_posix())
    else:
        config = load_config(root)
        print(json.dumps(_candidate_manifest(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
