"""Acquire and validate target-blind CLASH member stellar-mass multipoles.

The extractor has no route to the Item 1 gravity-response labels.  It turns public
Molino et al. photometric catalogs into a common stellar morphology grammar, then
checks that grammar against independent Chandra morphology before a separate module
is allowed to join any gravity labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .gravity_g1_pilot import _file_sha256, _load_json, _metric
from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_SCHEMA = "invariant-gravity-roadmap-item2-stellar-multipoles-config-1.0"
MANIFEST_SCHEMA = "invariant-gravity-item2-clash-stellar-multipoles-manifest-1.0"
CONFIG_PATH = "configs/gravity_item2_stellar_multipoles.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_item2_clash_stellar_multipoles.py"
MANIFEST_PATH = (
    "runs/gravity/roadmap/item-02-stellar-multipoles-v3-source/"
    "clash-stellar-multipoles-manifest.json"
)
FEATURE_PATH = (
    "runs/gravity/roadmap/item-02-stellar-multipoles-v3-source/clash-stellar-multipoles.tsv"
)

XRAY_NAME = {
    "a209": "A209",
    "a2261": "A2261",
    "a383": "A383",
    "a611": "A611",
    "macs0329": "0329-02",
    "macs0416": "0416-24",
    "macs0429": "0429-02",
    "macs0647": "0647+70",
    "macs0717": "0717+37",
    "macs0744": "0744+39",
    "macs1115": "1115+01",
    "macs1149": "1149+22",
    "macs1206": "1206-08",
    "macs1720": "1720+35",
    "macs1931": "1931-26",
    "ms2137": "MS2137",
    "rxj1347": "1347-1145",
    "rxj1532": "1532+30",
    "rxj2129": "2129+0005",
    "rxj2248": "2248-44",
}

FEATURE_COLUMNS = (
    "slug",
    "name",
    "cluster_z",
    "aperture_kpc",
    "weight_power",
    "catalog_rows",
    "member_count_including_bcg",
    "effective_member_count",
    "eight_sector_footprint_kpc",
    "concentration_c20",
    "centroid_shift",
    "quadrupole_amplitude",
    "m3_aperture_amplitude",
    "m4_aperture_amplitude",
    "multipole_energy",
    "position_angle_deg",
    "inner_50kpc_fraction",
    "inner_100kpc_fraction",
    "centroid_offset",
    "centered_quadrupole",
)


class GravityItem2ClashStellarMultipolesError(ValueError):
    """The target-blind CLASH stellar-multipole contract or evidence changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_envelope(value: Mapping[str, Any], *, label: str) -> None:
    body = dict(value)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityItem2ClashStellarMultipolesError(f"{label} content seal changed")


def load_config(root: Path) -> Mapping[str, Any]:
    """Validate only the target-blind portion of the frozen third-attempt contract."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityItem2ClashStellarMultipolesError("stellar multipole config schema changed")
    if config.get("status") != "exploratory_real_data_model_development":
        raise GravityItem2ClashStellarMultipolesError("stellar multipole status changed")
    roadmap = config.get("roadmap_binding", {})
    if (
        roadmap.get("item_number") != 2
        or roadmap.get("item_title") != "Shape and anisotropy"
        or _file_sha256(root / str(roadmap.get("path"))) != roadmap.get("file_sha256")
    ):
        raise GravityItem2ClashStellarMultipolesError("stellar multipole roadmap changed")
    authorization = config.get("authorization", {})
    if (
        authorization.get("network_acquisition_allowed") is not True
        or authorization.get("paid_model_calls_allowed") is not False
        or authorization.get("sparc_confirmation_evaluator_accesses_allowed") != 0
        or authorization.get("direct_lensing_likelihood_evaluations_allowed") != 0
        or authorization.get("sequential_G6_G7_G8_advanced") is not False
    ):
        raise GravityItem2ClashStellarMultipolesError("stellar multipole authorization changed")

    sources = config.get("sources", {})
    for key in ("galaxy_unwise_features", "clash_bcg_baryons", "clash_xray_morphology"):
        binding = sources.get(key, {})
        if _file_sha256(root / str(binding.get("path"))) != binding.get("file_sha256"):
            raise GravityItem2ClashStellarMultipolesError(f"{key} source changed")
    galaxy = sources["galaxy_unwise_features"]
    if _file_sha256(root / str(galaxy.get("manifest_path"))) != galaxy.get("manifest_sha256"):
        raise GravityItem2ClashStellarMultipolesError("unWISE manifest changed")
    if len(sources.get("clash_molino_catalogs", ())) != 20:
        raise GravityItem2ClashStellarMultipolesError("Molino source population changed")
    if {row["slug"] for row in sources["clash_molino_catalogs"]} != set(XRAY_NAME):
        raise GravityItem2ClashStellarMultipolesError("Molino cluster identities changed")

    extraction = config.get("target_blind_extraction", {})
    if (
        extraction.get("catalog_schema_columns") != 102
        or extraction.get("common_aperture_kpc") != 150
        or extraction.get("central_catalog_exclusion_kpc") != 5
        or extraction.get("weight_powers") != [0, 0.5, 1]
        or extraction.get("primary_weight_power") != 1
        or extraction.get("target_fields_available_to_feature_computation") is not False
        or extraction.get("lensing_corrected_columns_used") != []
        or extraction.get("gravity_response_receipts_read_by_extractor") != 0
    ):
        raise GravityItem2ClashStellarMultipolesError("stellar extraction boundary changed")
    selection = extraction.get("member_selection", {})
    if selection != {
        "point_source": 0,
        "maximum_sextractor_photoflag": 3,
        "minimum_observed_bands": 16,
        "minimum_detected_bands": 8,
        "minimum_signal_to_noise": 5,
        "stellar_mass_log10_solar_range": [6, 13],
        "secure_specz_maximum_delta_over_one_plus_z": 0.02,
        "photoz_interval_must_contain_cluster_z": True,
        "photoz_maximum_delta_over_one_plus_z": 0.06,
        "minimum_photoz_odds": 0.5,
    }:
        raise GravityItem2ClashStellarMultipolesError("member selection changed")
    return config


def parse_catalog(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        header = next(line for line in lines if line.startswith("# CLASHID"))
    except StopIteration as exc:
        raise GravityItem2ClashStellarMultipolesError("Molino header missing") from exc
    names = header[2:].split()
    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != len(names):
            raise GravityItem2ClashStellarMultipolesError(
                f"{path.name} has {len(values)} fields, expected {len(names)}"
            )
        rows.append(dict(zip(names, values, strict=True)))
    return names, rows


def _finite(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return math.isfinite(number) and number != -99.0


def is_member(row: Mapping[str, str], selection: Mapping[str, Any]) -> tuple[bool, str]:
    """Apply the frozen catalog-quality and redshift membership rule."""

    if int(row["PointS"]) != int(selection["point_source"]):
        return False, "point_source"
    if int(row["photoflag"]) > int(selection["maximum_sextractor_photoflag"]):
        return False, "photoflag"
    if int(row["nfobs"]) < int(selection["minimum_observed_bands"]) or int(row["nfdet"]) < int(
        selection["minimum_detected_bands"]
    ):
        return False, "band_coverage"
    if float(row["s2n"]) < float(selection["minimum_signal_to_noise"]):
        return False, "signal_to_noise"
    if not _finite(row["Stell_Mass"]):
        return False, "stellar_mass"
    low_mass, high_mass = selection["stellar_mass_log10_solar_range"]
    if not float(low_mass) <= float(row["Stell_Mass"]) <= float(high_mass):
        return False, "stellar_mass_range"

    cluster_z = float(row["clusterz"])
    spec_z = float(row["SpeczValue"])
    secure_spec = (
        spec_z > 0.0
        and int(row["SpeczQual"]) == 0
        and abs(spec_z - cluster_z)
        <= float(selection["secure_specz_maximum_delta_over_one_plus_z"]) * (1.0 + cluster_z)
    )
    photo_member = (
        _finite(row["zb_1"])
        and _finite(row["zb_Min_1"])
        and _finite(row["zb_Max_1"])
        and float(row["zb_Min_1"]) <= cluster_z <= float(row["zb_Max_1"])
        and abs(float(row["zb_1"]) - cluster_z)
        <= float(selection["photoz_maximum_delta_over_one_plus_z"]) * (1.0 + cluster_z)
        and float(row["Odds_1"]) >= float(selection["minimum_photoz_odds"])
    )
    if not (secure_spec or photo_member):
        return False, "redshift"
    return True, "member"


def _sector_footprint(rows: Sequence[Mapping[str, str]], sectors: int = 8) -> float:
    maxima = [0.0] * sectors
    for row in rows:
        # The two legacy BCG fields contain mosaic pixel coordinates despite their labels.
        dx = float(row["x"]) - float(row["BCG_pos_RA"])
        dy = float(row["y"]) - float(row["BCG_pos_Dec"])
        angle = math.atan2(dy, dx) % (2.0 * math.pi)
        sector = min(sectors - 1, int(angle / (2.0 * math.pi) * sectors))
        maxima[sector] = max(maxima[sector], 1000.0 * float(row["PhyDistBCG"]))
    return min(maxima)


def _read_bcg_baryons(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = root / str(config["sources"]["clash_bcg_baryons"]["path"])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 20 or {row["slug"] for row in rows} != set(XRAY_NAME):
        raise GravityItem2ClashStellarMultipolesError("BCG baryon population changed")
    return {
        row["slug"]: {
            "name": row["name"],
            "redshift": float(row["redshift"]),
            "mass_solar": 1.0e11 * float(row["bcg_stellar_mass_1e11_msun"]),
        }
        for row in rows
    }


def measure_catalog(
    rows: Sequence[Mapping[str, str]],
    *,
    bcg_mass_solar: float,
    aperture_kpc: float,
    central_exclusion_kpc: float,
    weight_power: float,
    selection: Mapping[str, Any],
) -> dict[str, float]:
    """Measure the common BCG-centered stellar multipole grammar."""

    points: list[tuple[float, float, float]] = [(0.0, 0.0, bcg_mass_solar)]
    for row in rows:
        if not is_member(row, selection)[0]:
            continue
        radius = 1000.0 * float(row["PhyDistBCG"])
        if radius <= central_exclusion_kpc or radius > aperture_kpc:
            continue
        dx_pixel = float(row["x"]) - float(row["BCG_pos_RA"])
        dy_pixel = float(row["y"]) - float(row["BCG_pos_Dec"])
        pixel_radius = math.hypot(dx_pixel, dy_pixel)
        if pixel_radius <= 0.0:
            continue
        points.append(
            (
                radius * dx_pixel / pixel_radius,
                radius * dy_pixel / pixel_radius,
                10.0 ** float(row["Stell_Mass"]),
            )
        )

    weights = [(mass / 1.0e10) ** weight_power for _, _, mass in points]
    total = sum(weights)
    centroid_x = sum(w * x for w, (x, _, _) in zip(weights, points, strict=True)) / total
    centroid_y = sum(w * y for w, (_, y, _) in zip(weights, points, strict=True)) / total
    centered = [(x - centroid_x, y - centroid_y) for x, y, _ in points]
    c_xx = sum(w * x * x for w, (x, _) in zip(weights, centered, strict=True))
    c_yy = sum(w * y * y for w, (_, y) in zip(weights, centered, strict=True))
    c_xy = sum(w * x * y for w, (x, y) in zip(weights, centered, strict=True))
    centered_q = math.hypot(c_xx - c_yy, 2.0 * c_xy) / max(c_xx + c_yy, 1.0e-30)
    position_angle = 0.5 * math.degrees(math.atan2(2.0 * c_xy, c_xx - c_yy))

    o_xx = sum(w * x * x for w, (x, _, _) in zip(weights, points, strict=True))
    o_yy = sum(w * y * y for w, (_, y, _) in zip(weights, points, strict=True))
    o_xy = sum(w * x * y for w, (x, y, _) in zip(weights, points, strict=True))
    quadrupole = math.hypot(o_xx - o_yy, 2.0 * o_xy) / max(o_xx + o_yy, 1.0e-30)

    centroid_points = []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        included = [
            (w, point)
            for w, point in zip(weights, points, strict=True)
            if math.hypot(point[0], point[1]) <= fraction * aperture_kpc
        ]
        subtotal = sum(w for w, _ in included)
        centroid_points.append(
            (
                sum(w * point[0] for w, point in included) / subtotal,
                sum(w * point[1] for w, point in included) / subtotal,
            )
        )
    mean_x = sum(point[0] for point in centroid_points) / len(centroid_points)
    mean_y = sum(point[1] for point in centroid_points) / len(centroid_points)
    centroid_shift = (
        math.sqrt(
            sum((x - mean_x) ** 2 + (y - mean_y) ** 2 for x, y in centroid_points)
            / len(centroid_points)
        )
        / aperture_kpc
    )

    moments = {}
    for degree in (3, 4):
        real = 0.0
        imaginary = 0.0
        for weight, (x, y, _) in zip(weights, points, strict=True):
            radius = math.hypot(x, y) / aperture_kpc
            angle = math.atan2(y, x)
            real += weight * radius**degree * math.cos(degree * angle)
            imaginary += weight * radius**degree * math.sin(degree * angle)
        moments[degree] = math.hypot(real, imaginary) / total

    def fraction_within(radius_kpc: float) -> float:
        return (
            sum(
                weight
                for weight, (x, y, _) in zip(weights, points, strict=True)
                if math.hypot(x, y) <= radius_kpc
            )
            / total
        )

    energy = math.sqrt(quadrupole**2 + moments[3] ** 2 + moments[4] ** 2)
    result = {
        "member_count_including_bcg": float(len(points)),
        "effective_member_count": total * total / sum(weight * weight for weight in weights),
        "concentration_c20": fraction_within(0.2 * aperture_kpc),
        "centroid_shift": centroid_shift,
        "quadrupole_amplitude": quadrupole,
        "m3_aperture_amplitude": moments[3],
        "m4_aperture_amplitude": moments[4],
        "multipole_energy": energy,
        "position_angle_deg": position_angle,
        "inner_50kpc_fraction": fraction_within(50.0),
        "inner_100kpc_fraction": fraction_within(100.0),
        "centroid_offset": math.hypot(centroid_x, centroid_y) / aperture_kpc,
        "centered_quadrupole": centered_q,
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise GravityItem2ClashStellarMultipolesError("non-finite stellar multipole")
    return result


def _read_xray(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    path = root / str(config["sources"]["clash_xray_morphology"]["path"])
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(
            (line for line in handle if not line.startswith("#")), delimiter="\t"
        )
        for row in reader:
            rows[str(row["source_name"])] = {
                "concentration": float(row["concentration"]),
                "centroid_shift": float(row["centroid_shift"]),
                "p30": float(row["p30"]),
                "p40": float(row["p40"]),
                "ellipticity": 1.0 - float(row["axis_ratio"]),
                "position_angle_deg": float(row["position_angle_deg"]),
            }
    return rows


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_scale == 0.0 or right_scale == 0.0:
        raise GravityItem2ClashStellarMultipolesError("constant validation vector")
    return numerator / (left_scale * right_scale)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def external_validation(
    rows: Sequence[Mapping[str, Any]],
    xray: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
    permutations: int,
) -> dict[str, Any]:
    comparisons = (
        ("concentration_c20", "concentration", False),
        ("centroid_shift", "centroid_shift", False),
        ("quadrupole_amplitude", "ellipticity", False),
        ("m3_aperture_amplitude", "p30", True),
        ("m4_aperture_amplitude", "p40", True),
    )
    components = {}
    feature_ranks = []
    target_ranks = []
    for feature, target, logarithmic in comparisons:
        left = [float(row[feature]) for row in rows]
        right = [
            math.log10(float(xray[XRAY_NAME[str(row["slug"])]][target]))
            if logarithmic
            else float(xray[XRAY_NAME[str(row["slug"])]][target])
            for row in rows
        ]
        correlation = _spearman(left, right)
        components[f"{feature}_vs_xray_{target}"] = _metric(correlation)
        feature_ranks.append(_rank(left))
        target_ranks.append(_rank(right))
    observed = sum(
        _pearson(left, right) for left, right in zip(feature_ranks, target_ranks, strict=True)
    ) / len(comparisons)
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    exceedances = 0
    for _ in range(permutations):
        rng.shuffle(indices)
        score = sum(
            _pearson(left, [right[index] for index in indices])
            for left, right in zip(feature_ranks, target_ranks, strict=True)
        ) / len(comparisons)
        exceedances += score >= observed

    angle_differences = []
    for row in rows:
        stellar = float(row["position_angle_deg"])
        gas = float(xray[XRAY_NAME[str(row["slug"])]]["position_angle_deg"])
        angle_differences.append(abs((stellar - gas + 90.0) % 180.0 - 90.0))
    ordered_angles = sorted(angle_differences)
    return {
        "components": components,
        "joint": {
            "statistic": "mean_of_five_spearman_correlations",
            "observed": _metric(observed),
            "whole_cluster_vector_permuted": True,
            "permutations": permutations,
            "seed": seed,
            "one_sided_p_value": _metric((exceedances + 1.0) / (permutations + 1.0)),
        },
        "position_axis": {
            "median_absolute_difference_deg": _metric(
                0.5 * (ordered_angles[9] + ordered_angles[10])
            ),
            "fraction_within_30_deg": _metric(
                sum(value <= 30.0 for value in angle_differences) / len(angle_differences)
            ),
        },
    }


def _payload(row: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for column in FEATURE_COLUMNS:
        value = row[column]
        if isinstance(value, float):
            result[column] = format(value, ".15e")
        else:
            result[column] = str(value)
    return result


def _render_features(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FEATURE_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_payload(row))
    return buffer.getvalue().encode("utf-8")


def _source_url(config: Mapping[str, Any], slug: str) -> str:
    return str(config["sources"]["molino_url_template"]).format(slug=slug)


def acquire_features(
    root: Path,
    *,
    cache_dir: Path,
    feature_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Acquire exact catalog bytes and write the target-blind representation receipt."""

    root = root.resolve()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(root)
    extraction = config["target_blind_extraction"]
    selection = extraction["member_selection"]
    bcg = _read_bcg_baryons(root, config)
    all_rows = []
    source_records = []
    for source in config["sources"]["clash_molino_catalogs"]:
        slug = str(source["slug"])
        path = cache_dir / f"hlsp_clash_hst_ir_{slug}_cat-molino.txt"
        if not path.exists():
            request = urllib.request.Request(
                _source_url(config, slug),
                headers={"User-Agent": "Invariant target-blind source extraction/1.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read()
            temporary = path.with_suffix(".txt.tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
        if _sha256(path) != source["sha256"]:
            raise GravityItem2ClashStellarMultipolesError(f"{slug} catalog bytes changed")
        names, catalog = parse_catalog(path)
        if len(names) != int(extraction["catalog_schema_columns"]):
            raise GravityItem2ClashStellarMultipolesError(f"{slug} schema changed")
        footprint = _sector_footprint(catalog)
        for power in extraction["weight_powers"]:
            measured = measure_catalog(
                catalog,
                bcg_mass_solar=float(bcg[slug]["mass_solar"]),
                aperture_kpc=float(extraction["common_aperture_kpc"]),
                central_exclusion_kpc=float(extraction["central_catalog_exclusion_kpc"]),
                weight_power=float(power),
                selection=selection,
            )
            all_rows.append(
                {
                    "slug": slug,
                    "name": bcg[slug]["name"],
                    "cluster_z": float(catalog[0]["clusterz"]),
                    "aperture_kpc": float(extraction["common_aperture_kpc"]),
                    "weight_power": float(power),
                    "catalog_rows": len(catalog),
                    "eight_sector_footprint_kpc": footprint,
                    **measured,
                }
            )
        source_records.append(
            {
                "slug": slug,
                "url": _source_url(config, slug),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "catalog_rows": len(catalog),
                "schema_columns": len(names),
                "eight_sector_footprint_kpc": _metric(footprint),
            }
        )

    feature_bytes = _render_features(all_rows)
    feature_path = (feature_path or root / FEATURE_PATH).resolve()
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_bytes(feature_bytes)
    xray = _read_xray(root, config)
    gate = config["representation_validation_gate"]
    validation = {}
    for power in extraction["weight_powers"]:
        power_rows = [row for row in all_rows if float(row["weight_power"]) == float(power)]
        validation[str(power)] = external_validation(
            power_rows,
            xray,
            seed=int(gate["seed"]),
            permutations=int(gate["permutations"]),
        )
    primary = validation[str(extraction["primary_weight_power"])]
    primary_correlations = [float(value) for value in primary["components"].values()]
    primary_counts = sorted(
        int(row["member_count_including_bcg"])
        for row in all_rows
        if float(row["weight_power"]) == float(extraction["primary_weight_power"])
    )
    checks = {
        "all_catalogs_exact_and_complete": len(source_records) == int(gate["expected_catalogs"]),
        "common_aperture_covered_in_eight_sectors": min(
            float(row["eight_sector_footprint_kpc"]) for row in source_records
        )
        >= float(gate["minimum_eight_sector_footprint_kpc"]),
        "minimum_members_present": min(primary_counts)
        >= int(gate["minimum_members_including_bcg_per_cluster"]),
        "median_members_present": 0.5 * (primary_counts[9] + primary_counts[10])
        >= int(gate["minimum_median_members_including_bcg"]),
        "all_primary_morphology_directions_positive": all(
            value > 0.0 for value in primary_correlations
        ),
        "minimum_strong_primary_correlations": sum(value > 0.2 for value in primary_correlations)
        >= int(gate["minimum_primary_correlations_above_0p2"]),
        "joint_primary_permutation_pass": float(primary["joint"]["one_sided_p_value"])
        <= float(gate["maximum_joint_whole_cluster_permutation_p"]),
        "position_axis_median_pass": float(
            primary["position_axis"]["median_absolute_difference_deg"]
        )
        <= float(gate["maximum_median_position_axis_difference_deg"]),
        "position_axis_fraction_pass": float(primary["position_axis"]["fraction_within_30_deg"])
        >= float(gate["minimum_fraction_position_axis_within_30_deg"]),
        "weighting_robustness_pass": all(
            float(validation[str(power)]["joint"]["observed"]) > 0.0
            and float(validation[str(power)]["joint"]["one_sided_p_value"])
            <= float(gate["maximum_robustness_joint_permutation_p"])
            for power in gate["robustness_weight_powers"]
        ),
        "gravity_response_fields_absent": True,
        "lensing_corrected_mass_fields_absent": True,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "goal": "TARGET_BLIND_CLASH_STELLAR_MULTIPOLE_REPRESENTATION",
        "decision": "PASS_TARGET_BLIND_REPRESENTATION_GATE"
        if all(checks.values())
        else "REJECT_TARGET_BLIND_REPRESENTATION_GATE",
        "counts": {
            "catalogs": len(source_records),
            "feature_rows": len(all_rows),
            "primary_clusters": len(primary_counts),
            "minimum_primary_members_including_bcg": min(primary_counts),
            "median_primary_members_including_bcg": _metric(
                0.5 * (primary_counts[9] + primary_counts[10])
            ),
            "gravity_response_receipts_read": 0,
            "lensing_corrected_fields_used": 0,
        },
        "checks": checks,
        "external_xray_validation": validation,
        "feature_file": {
            "path": feature_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(feature_bytes).hexdigest(),
        },
        "source_records": source_records,
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "sha256": _sha256(root / CONFIG_PATH)},
            "extractor": {"path": SOURCE_PATH, "sha256": _sha256(root / SOURCE_PATH)},
            "bcg_baryons": {
                "path": config["sources"]["clash_bcg_baryons"]["path"],
                "sha256": config["sources"]["clash_bcg_baryons"]["file_sha256"],
                "raw_source_sha256": config["sources"]["clash_bcg_baryons"]["raw_source_sha256"],
            },
            "xray_morphology": {
                "path": config["sources"]["clash_xray_morphology"]["path"],
                "sha256": config["sources"]["clash_xray_morphology"]["file_sha256"],
            },
        },
        "limitations": [
            "The HST catalogs cover cluster cores, so the common aperture is 150 kpc rather than the 500 kpc X-ray validation aperture.",
            "Photometric-redshift membership is probabilistic and residual foreground/background contamination can remain.",
            "Member stellar mass omits intracluster light and hot gas and is not a complete baryonic mass map.",
            "Agreement with X-ray morphology validates a representation; it does not validate a gravity law.",
        ],
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    manifest_path = (manifest_path or root / MANIFEST_PATH).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def validate_extraction(root: Path, *, cache_dir: Path | None = None) -> Mapping[str, Any]:
    root = root.resolve()
    config = load_config(root)
    manifest = _load_json(root / MANIFEST_PATH)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise GravityItem2ClashStellarMultipolesError("stellar manifest schema changed")
    _verify_envelope(manifest, label="stellar manifest")
    if manifest.get("decision") != "PASS_TARGET_BLIND_REPRESENTATION_GATE":
        raise GravityItem2ClashStellarMultipolesError("stellar representation did not pass")
    if not all(manifest.get("checks", {}).values()):
        raise GravityItem2ClashStellarMultipolesError("stellar representation gate changed")
    feature = manifest.get("feature_file", {})
    if _sha256(root / str(feature.get("path"))) != feature.get("sha256"):
        raise GravityItem2ClashStellarMultipolesError("stellar feature file changed")
    bindings = manifest.get("source_bindings", {})
    if (
        _sha256(root / CONFIG_PATH) != bindings.get("config", {}).get("sha256")
        or _sha256(root / SOURCE_PATH) != bindings.get("extractor", {}).get("sha256")
        or _sha256(root / str(bindings.get("bcg_baryons", {}).get("path")))
        != bindings.get("bcg_baryons", {}).get("sha256")
        or _sha256(root / str(bindings.get("xray_morphology", {}).get("path")))
        != bindings.get("xray_morphology", {}).get("sha256")
    ):
        raise GravityItem2ClashStellarMultipolesError("stellar source binding changed")
    if (
        manifest.get("counts", {}).get("catalogs") != 20
        or manifest.get("counts", {}).get("feature_rows") != 60
        or manifest.get("counts", {}).get("gravity_response_receipts_read") != 0
        or manifest.get("counts", {}).get("lensing_corrected_fields_used") != 0
    ):
        raise GravityItem2ClashStellarMultipolesError("stellar extraction counts changed")
    if cache_dir is not None:
        expected = {
            row["slug"]: row["sha256"] for row in config["sources"]["clash_molino_catalogs"]
        }
        for slug, digest in expected.items():
            path = cache_dir.resolve() / f"hlsp_clash_hst_ir_{slug}_cat-molino.txt"
            if not path.exists() or _sha256(path) != digest:
                raise GravityItem2ClashStellarMultipolesError(f"cached {slug} catalog changed")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--root", type=Path, default=Path.cwd())
    acquire.add_argument("--cache-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "acquire":
        result = acquire_features(args.root, cache_dir=args.cache_dir)
    else:
        result = validate_extraction(args.root, cache_dir=args.cache_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
