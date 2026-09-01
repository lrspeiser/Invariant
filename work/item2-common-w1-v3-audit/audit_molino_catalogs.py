"""Target-blind source audit for CLASH Molino stellar-mass catalogs.

This scratch program is deliberately forbidden from reading any Item 1 gravity-response
receipt.  It only acquires public photometric catalogs and reports catalog/member geometry
needed to decide whether a preregistered third Item 2 experiment is defensible.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import urllib.request
import csv


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "item2-common-w1-v3-audit" / "molino-raw"

SLUGS = (
    "a209",
    "a2261",
    "a383",
    "a611",
    "macs0329",
    "macs0416",
    "macs0429",
    "macs0647",
    "macs0717",
    "macs0744",
    "macs1115",
    "macs1149",
    "macs1206",
    "macs1720",
    "macs1931",
    "ms2137",
    "rxj1347",
    "rxj1532",
    "rxj2129",
    "rxj2248",
)

BASE = "https://archive.stsci.edu/missions/hlsp/clash/{slug}/catalogs/molino/"
NAME = "hlsp_clash_hst_ir_{slug}_cat-molino.txt"

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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def acquire() -> list[dict[str, object]]:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt: list[dict[str, object]] = []
    for slug in SLUGS:
        filename = NAME.format(slug=slug)
        url = BASE.format(slug=slug) + filename
        path = OUT / filename
        if path.exists():
            raw = path.read_bytes()
        else:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Invariant target-blind source audit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read()
            path.write_bytes(raw)
        receipt.append(
            {
                "slug": slug,
                "url": url,
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
        )
    return receipt


def parse(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    header = next(line for line in lines if line.startswith("# CLASHID"))
    names = header[2:].split()
    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != len(names):
            raise ValueError(f"{path.name}: {len(values)} fields, expected {len(names)}")
        rows.append(dict(zip(names, values, strict=True)))
    return names, rows


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) != -99.0
    except ValueError:
        return False


def is_member(row: dict[str, str]) -> tuple[bool, str]:
    """Catalog-quality and redshift rule frozen without gravity-response labels."""

    if int(row["PointS"]) != 0:
        return False, "point_source"
    # SExtractor bits 1 and 2 (neighbour/blend) are tolerated in crowded cluster cores;
    # bit 4 and higher flag saturation, truncation, or corrupt/incomplete photometry.
    if int(row["photoflag"]) > 3:
        return False, "photoflag"
    if int(row["nfobs"]) < 16 or int(row["nfdet"]) < 8:
        return False, "band_coverage"
    if float(row["s2n"]) < 5.0:
        return False, "signal_to_noise"
    if not finite(row["Stell_Mass"]):
        return False, "stellar_mass"
    log_mass = float(row["Stell_Mass"])
    if not 6.0 <= log_mass <= 13.0:
        return False, "stellar_mass_range"

    cluster_z = float(row["clusterz"])
    spec_z = float(row["SpeczValue"])
    spec_quality = int(row["SpeczQual"])
    secure_spec = (
        spec_z > 0.0
        and spec_quality == 0
        and abs(spec_z - cluster_z) <= 0.02 * (1.0 + cluster_z)
    )

    photo_z = float(row["zb_1"])
    z_low = float(row["zb_Min_1"])
    z_high = float(row["zb_Max_1"])
    photo_member = (
        finite(row["zb_1"])
        and finite(row["zb_Min_1"])
        and finite(row["zb_Max_1"])
        and z_low <= cluster_z <= z_high
        and abs(photo_z - cluster_z) <= 0.06 * (1.0 + cluster_z)
        and float(row["Odds_1"]) >= 0.5
    )
    if not (secure_spec or photo_member):
        return False, "redshift"
    return True, "member"


def sector_complete_radius(rows: list[dict[str, str]], sectors: int = 8) -> float:
    """Conservative radial footprint: the least-populated angular sector's outer reach."""

    maxima = [0.0] * sectors
    for row in rows:
        # Despite the legacy column descriptions saying RA/Dec, these two BCG fields
        # contain mosaic pixel coordinates (for example 2315, 2704 in A383).  Pair them
        # with the catalog x/y values; mixing them with sky degrees makes every footprint
        # diagnostic meaningless.
        dx = float(row["x"]) - float(row["BCG_pos_RA"])
        dy = float(row["y"]) - float(row["BCG_pos_Dec"])
        angle = math.atan2(dy, dx) % (2.0 * math.pi)
        sector = min(sectors - 1, int(angle / (2.0 * math.pi) * sectors))
        maxima[sector] = max(maxima[sector], 1000.0 * float(row["PhyDistBCG"]))
    return min(maxima)


def read_allowed_bcg_masses() -> dict[str, float]:
    """Read only AName and baryonic BCG M* from the target-free portion of Table 1."""

    path = ROOT / "work" / "item2-common-w1-v3-audit" / "clash-table1.tsv"
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("recno\t"))
    names = lines[header_index].split("\t")
    rows: dict[str, float] = {}
    for values in csv.reader(lines[header_index + 1 :], delimiter="\t"):
        if not values or not values[0].strip().isdigit():
            continue
        row = dict(zip(names, values, strict=False))
        rows[row["AName"].strip().lower()] = 1.0e11 * float(row["M*"].strip())
    if set(rows) != set(SLUGS):
        raise ValueError(f"BCG mass rows differ from requested clusters: {set(rows) ^ set(SLUGS)}")
    return rows


def read_xray_morphology() -> dict[str, dict[str, float]]:
    path = (
        ROOT
        / "runs"
        / "gravity"
        / "roadmap"
        / "item-02-shape-anisotropy-v1-source"
        / "clash_xray_morphology_500kpc.tsv"
    )
    rows: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
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


def rank(values: list[float]) -> list[float]:
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


def pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(rank(left), rank(right))


def point_cloud_features(
    rows: list[dict[str, str]], bcg_mass: float, *, aperture_kpc: float, mass_power: float
) -> dict[str, float]:
    points: list[tuple[float, float, float]] = [(0.0, 0.0, bcg_mass)]
    for row in rows:
        if not is_member(row)[0]:
            continue
        radius = 1000.0 * float(row["PhyDistBCG"])
        if radius <= 5.0 or radius > aperture_kpc:
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

    weights = [(mass / 1.0e10) ** mass_power for _, _, mass in points]
    total = sum(weights)
    centroid_x = sum(weight * point[0] for weight, point in zip(weights, points, strict=True)) / total
    centroid_y = sum(weight * point[1] for weight, point in zip(weights, points, strict=True)) / total
    centered = [(x - centroid_x, y - centroid_y) for x, y, _ in points]
    i_xx = sum(weight * x * x for weight, (x, _) in zip(weights, centered, strict=True))
    i_yy = sum(weight * y * y for weight, (_, y) in zip(weights, centered, strict=True))
    i_xy = sum(weight * x * y for weight, (x, y) in zip(weights, centered, strict=True))
    quadrupole = math.hypot(i_xx - i_yy, 2.0 * i_xy) / max(i_xx + i_yy, 1.0e-30)
    position_angle = 0.5 * math.degrees(math.atan2(2.0 * i_xy, i_xx - i_yy))
    origin_i_xx = sum(weight * x * x for weight, (x, _, _) in zip(weights, points, strict=True))
    origin_i_yy = sum(weight * y * y for weight, (_, y, _) in zip(weights, points, strict=True))
    origin_i_xy = sum(weight * x * y for weight, (x, y, _) in zip(weights, points, strict=True))
    bcg_quadrupole = math.hypot(
        origin_i_xx - origin_i_yy, 2.0 * origin_i_xy
    ) / max(origin_i_xx + origin_i_yy, 1.0e-30)

    aperture_centroids = []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        included = [
            (weight, point)
            for weight, point in zip(weights, points, strict=True)
            if math.hypot(point[0], point[1]) <= fraction * aperture_kpc
        ]
        included_weight = sum(weight for weight, _ in included)
        aperture_centroids.append(
            (
                sum(weight * point[0] for weight, point in included) / included_weight,
                sum(weight * point[1] for weight, point in included) / included_weight,
            )
        )
    mean_centroid_x = sum(point[0] for point in aperture_centroids) / len(aperture_centroids)
    mean_centroid_y = sum(point[1] for point in aperture_centroids) / len(aperture_centroids)
    centroid_shift_multiscale = math.sqrt(
        sum(
            (x - mean_centroid_x) ** 2 + (y - mean_centroid_y) ** 2
            for x, y in aperture_centroids
        )
        / len(aperture_centroids)
    ) / aperture_kpc

    complex_moments: dict[int, float] = {}
    for degree in (2, 3, 4):
        real = 0.0
        imaginary = 0.0
        for weight, (x, y, _) in zip(weights, points, strict=True):
            radius = math.hypot(x, y) / aperture_kpc
            angle = math.atan2(y, x)
            real += weight * radius**degree * math.cos(degree * angle)
            imaginary += weight * radius**degree * math.sin(degree * angle)
        complex_moments[degree] = math.hypot(real, imaginary) / total

    return {
        "member_count_including_bcg": float(len(points)),
        "effective_member_count": total * total / sum(weight * weight for weight in weights),
        "centroid_offset": math.hypot(centroid_x, centroid_y) / aperture_kpc,
        "centroid_shift_multiscale": centroid_shift_multiscale,
        "centered_quadrupole": quadrupole,
        "bcg_quadrupole": bcg_quadrupole,
        "position_angle_deg": position_angle,
        "m2_aperture": complex_moments[2],
        "m3_aperture": complex_moments[3],
        "m4_aperture": complex_moments[4],
        "inner_30kpc_fraction": sum(
            weight
            for weight, (x, y, _) in zip(weights, points, strict=True)
            if math.hypot(x, y) <= 30.0
        )
        / total,
        "inner_50kpc_fraction": sum(
            weight
            for weight, (x, y, _) in zip(weights, points, strict=True)
            if math.hypot(x, y) <= 50.0
        )
        / total,
        "inner_100kpc_fraction": sum(
            weight
            for weight, (x, y, _) in zip(weights, points, strict=True)
            if math.hypot(x, y) <= 100.0
        )
        / total,
    }


def external_validation(feature_rows: list[dict[str, object]]) -> dict[str, object]:
    xray = read_xray_morphology()
    comparisons = {
        "inner_30kpc_fraction_vs_xray_concentration": ("inner_30kpc_fraction", "concentration"),
        "inner_50kpc_fraction_vs_xray_concentration": ("inner_50kpc_fraction", "concentration"),
        "inner_100kpc_fraction_vs_xray_concentration": ("inner_100kpc_fraction", "concentration"),
        "centroid_offset_vs_xray_centroid_shift": ("centroid_offset", "centroid_shift"),
        "centroid_shift_multiscale_vs_xray_centroid_shift": (
            "centroid_shift_multiscale",
            "centroid_shift",
        ),
        "centered_quadrupole_vs_xray_ellipticity": ("centered_quadrupole", "ellipticity"),
        "bcg_quadrupole_vs_xray_ellipticity": ("bcg_quadrupole", "ellipticity"),
        "m3_aperture_vs_log_xray_p30": ("m3_aperture", "p30"),
        "m4_aperture_vs_log_xray_p40": ("m4_aperture", "p40"),
    }
    results: dict[str, object] = {}
    for name, (feature, target) in comparisons.items():
        left = [float(row[feature]) for row in feature_rows]
        right = [
            math.log10(xray[XRAY_NAME[str(row["slug"])]][target])
            if target in {"p30", "p40"}
            else xray[XRAY_NAME[str(row["slug"])]][target]
            for row in feature_rows
        ]
        results[name] = {"objects": len(left), "spearman": spearman(left, right)}

    angle_differences = []
    for row in feature_rows:
        stellar = float(row["position_angle_deg"])
        gas_row = xray[XRAY_NAME[str(row["slug"])]]
        gas = gas_row["position_angle_deg"]
        difference = abs((stellar - gas + 90.0) % 180.0 - 90.0)
        angle_differences.append(difference)
    results["stellar_vs_xray_position_angle"] = {
        "objects": len(angle_differences),
        "median_absolute_axis_difference_deg": sorted(angle_differences)[len(angle_differences) // 2],
        "fraction_within_30_deg": sum(value <= 30.0 for value in angle_differences)
        / len(angle_differences),
    }

    primary_pairs = (
        ("inner_30kpc_fraction", "concentration", False),
        ("centroid_shift_multiscale", "centroid_shift", False),
        ("bcg_quadrupole", "ellipticity", False),
        ("m3_aperture", "p30", True),
        ("m4_aperture", "p40", True),
    )
    feature_ranks = [
        rank([float(row[feature]) for row in feature_rows])
        for feature, _, _ in primary_pairs
    ]
    target_ranks = [
        rank(
            [
                math.log10(xray[XRAY_NAME[str(row["slug"])]] [target])
                if log_target
                else xray[XRAY_NAME[str(row["slug"])]] [target]
                for row in feature_rows
            ]
        )
        for _, target, log_target in primary_pairs
    ]
    observed_components = [
        pearson(left, right) for left, right in zip(feature_ranks, target_ranks, strict=True)
    ]
    if any(value is None for value in observed_components):
        raise ValueError("primary validation statistic includes a constant vector")
    observed = sum(float(value) for value in observed_components) / len(observed_components)
    rng = random.Random(20260827)
    permutations = 100_000
    exceedances = 0
    indices = list(range(len(feature_rows)))
    for _ in range(permutations):
        rng.shuffle(indices)
        score = sum(
            float(pearson(left, [right[index] for index in indices]))
            for left, right in zip(feature_ranks, target_ranks, strict=True)
        ) / len(primary_pairs)
        exceedances += score >= observed
    results["joint_five_metric_permutation"] = {
        "statistic": "mean_of_five_spearman_correlations",
        "observed": observed,
        "permutations": permutations,
        "seed": 20260827,
        "whole_cluster_vector_permuted": True,
        "one_sided_p_value": (exceedances + 1.0) / (permutations + 1.0),
    }
    return results


def summarize(source_rows: list[dict[str, object]]) -> dict[str, object]:
    clusters = []
    feature_sets: dict[str, list[dict[str, object]]] = {}
    bcg_masses = read_allowed_bcg_masses()
    apertures = (100.0, 150.0, 200.0, 250.0, 300.0, 500.0)
    for source in source_rows:
        path = ROOT / str(source["path"])
        names, rows = parse(path)
        selected = [row for row in rows if is_member(row)[0]]
        nearest = min(rows, key=lambda row: float(row["PhyDistBCG"]))
        counts = {
            f"within_{int(aperture)}_kpc": sum(
                1000.0 * float(row["PhyDistBCG"]) <= aperture for row in selected
            )
            for aperture in apertures
        }
        mass_counts = {
            f"within_{int(aperture)}_kpc": sum(
                10.0 ** float(row["Stell_Mass"])
                for row in selected
                if 1000.0 * float(row["PhyDistBCG"]) <= aperture
            )
            for aperture in apertures
        }
        failures: dict[str, int] = {}
        for row in rows:
            _, reason = is_member(row)
            failures[reason] = failures.get(reason, 0) + 1
        clusters.append(
            {
                "slug": source["slug"],
                "schema_columns": len(names),
                "catalog_rows": len(rows),
                "cluster_z": float(rows[0]["clusterz"]),
                "selected_members_all_radii": len(selected),
                "selected_counts": counts,
                "selected_stellar_mass_solar": mass_counts,
                "eight_sector_complete_radius_kpc": sector_complete_radius(rows),
                "nearest_detection_kpc": 1000.0 * float(nearest["PhyDistBCG"]),
                "nearest_detection_selected": is_member(nearest)[0],
                "nearest_detection_log_stellar_mass": (
                    float(nearest["Stell_Mass"]) if finite(nearest["Stell_Mass"]) else None
                ),
                "selection_outcomes": dict(sorted(failures.items())),
            }
        )
        for aperture in (100.0, 150.0):
            for mass_power in (0.0, 0.5, 1.0):
                key = f"r{int(aperture)}_mass_power_{mass_power:g}"
                features: dict[str, object] = {
                    "slug": source["slug"],
                    **point_cloud_features(
                        rows,
                        bcg_masses[str(source["slug"])],
                        aperture_kpc=aperture,
                        mass_power=mass_power,
                    ),
                }
                feature_sets.setdefault(key, []).append(features)
    return {
        "audit_boundary": {
            "gravity_response_receipts_read": 0,
            "lensing_mass_columns_used": [],
            "lensing_corrected_columns_used": [],
            "target_blind": True,
        },
        "source_count": len(source_rows),
        "sources": source_rows,
        "clusters": clusters,
        "feature_sets": {
            key: {
                "rows": rows,
                "external_xray_validation": external_validation(rows),
            }
            for key, rows in sorted(feature_sets.items())
        },
    }


def main() -> None:
    result = summarize(acquire())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
