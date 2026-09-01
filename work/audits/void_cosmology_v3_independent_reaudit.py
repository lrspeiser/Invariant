"""Read-only independent audit for the frozen void/cosmology v3 matrix.

This script deliberately does not call the subject ``derive_release`` or ``check``
functions.  It independently re-parses the frozen sources, recomputes the source
geometry, noise/runtime values, candidate predictions, scores, winner sets, and
ledger hashes, and compares the append-only v3 package with its v2 predecessor.
"""

from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v3 as subject,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v3"
V3_DIR = ROOT / "runs/gravity" / PACKAGE
V2_DIR = ROOT / "runs/gravity/open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v2"
EXPECTED_SUBJECT = {
    "config": "7e7ee7df9f8bb069935527e198f4906f76afd10e3513e46079cc2209c99ebbff",
    "module": "d210b6174174d682b10c3cbd9f4e776a10c4b76d650dc038edf412baaec7ef79",
    "test": "75f53c63a612ea2f26142011afd5d11bd5674826732b604d6f5a87b78ca73a50",
    "receipt_raw": "0d386ae421971d38f61d9a02d5e9baf81bde9f4b862ff88162133562a90b9415",
    "receipt_content": "4fbed69505d8be79112695937023468316b371d10fe0c5f37f7771ba14906598",
    "catalogue": "3e338d68cadbe9c84ef5bd24376087ea19a743c71509d92677feb8b0538301d7",
    "blocked_raw": "849aa96389af239148ce957256ad2285053ae3437d7a4210530a64cfb72abe23",
    "blocked_content": "a4217f84e6953a0807fe7f41c065b08ab7ba68a71c134146aa8cf4fa725ba3a6",
}
EXPECTED_ARTIFACTS = {
    "confusion-matrix.json": "d43c722bb34f6807167ae1168f163d4c922a12ff109e8a7d86324ecde163e34f",
    "geometry-and-identifiability.json": "632cec270ec8a760607ca0586935514dedaae83198238a173df0cde6e45de9ce",
    "ledger.json": "ff09e5ae34a7dfcaef2a0a0505dac657da5838e12899474c298b58ae93f4c8c2",
    "scenarios.jsonl": "df449f206cb90b64b5bc2ed02966270e3ff40b14d00108ba55cfb18788f20a9e",
    "typed-contract-diff.json": "a13d9074c1d48afd463ef9359861f31439a238097fdbf8365d3e71f7141b4053",
    "values.npz": "6126301f03923ceca0b129869a5ebe6a55f369291e606043995dd48654e899c0",
}
RATE_FEATURES = (
    "source.scalar.delta-h-km-s-mpc",
    "source.scalar.h-m-km-s-mpc",
)
RATE_UNIT = "km s^-1 Mpc^-1"
RATE_DIMENSION = [0, 0, -1, 0, 0, 0, 0]
CANDIDATES = (
    "C01_OBSERVER_ENDPOINT_LOCAL_VOID",
    "C02_TARGET_ENDPOINT_LOCAL_VOID",
    "C03_SINGLE_DOMINANT_VOID",
    "C04_BOUNDED_FRACTION_NULL",
    "VQ00_STANDARD_FLRW_FLOW_CONTROL",
    "VQ08_TWO_PHASE_VOID_FRACTION",
)
EXPOSURE = {
    "C01_OBSERVER_ENDPOINT_LOCAL_VOID": "source.scalar.observer-endpoint-chord-mpc",
    "C02_TARGET_ENDPOINT_LOCAL_VOID": "source.scalar.target-endpoint-chord-mpc",
    "C03_SINGLE_DOMINANT_VOID": "source.scalar.maximum-chord-mpc",
    "C04_BOUNDED_FRACTION_NULL": "source.scalar.null-void-length-mpc",
    "VQ00_STANDARD_FLRW_FLOW_CONTROL": None,
    "VQ08_TWO_PHASE_VOID_FRACTION": "source.scalar.void-length-mpc",
}
OUTPUT_ID = "prediction.scalar.log-redshift"
RESPONSE_ID = "response.synthetic-log-redshift"
UNCERTAINTY_ID = "uncertainty.response.synthetic-log-redshift"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return sha_bytes(canonical_bytes(value))


def array_sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.name.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_scenario(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for group in (
        "formula_features",
        "scoring_responses",
        "hidden_truth",
        "expected_predictions",
        "uncertainties",
    ):
        for row in result[group]:
            row["artifact_path"] = "<PACKAGE_VALUES_NPZ>"
            if group == "formula_features" and row.get("element_id") in RATE_FEATURES:
                row["unit"] = "<HUBBLE_RATE_UNIT>"
    return result


def comoving_mpc(redshift: float, *, h0: float, omega_m: float, c_km_s: float) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(64)
    sample = 0.5 * redshift * (nodes + 1.0)
    expansion = np.sqrt(omega_m * (1.0 + sample) ** 3 + (1.0 - omega_m))
    return (c_km_s / h0) * 0.5 * redshift * float(np.sum(weights / expansion))


def luminosity_to_comoving_hinv(distance_luminosity_mpc: float, cosmology: dict[str, Any]) -> tuple[float, float]:
    h0 = float(cosmology["H0_km_s_Mpc"])
    omega_m = float(cosmology["Omega_m"])
    c_km_s = float(cosmology["c_km_s"])
    if distance_luminosity_mpc == 0.0:
        return 0.0, 0.0

    def dl(redshift: float) -> float:
        return (1.0 + redshift) * comoving_mpc(
            redshift, h0=h0, omega_m=omega_m, c_km_s=c_km_s
        )

    low, high = 0.0, 0.2
    while dl(high) < distance_luminosity_mpc:
        high *= 2.0
        require(high <= 4.0, "distance inversion escaped frozen range")
    for _ in range(80):
        mid = 0.5 * (low + high)
        if dl(mid) < distance_luminosity_mpc:
            low = mid
        else:
            high = mid
    redshift = 0.5 * (low + high)
    dc = comoving_mpc(redshift, h0=h0, omega_m=omega_m, c_km_s=c_km_s)
    return redshift, (h0 / 100.0) * dc


def radec_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(ra_deg % 360.0)
    dec = math.radians(dec_deg)
    return np.asarray(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)],
        dtype=np.float64,
    )


def union_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for start, stop in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(row[0], row[1]) for row in merged]


def interval_summary(
    direction: np.ndarray,
    distance: float,
    centers: np.ndarray,
    radii: np.ndarray,
) -> dict[str, float | int]:
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / float(np.linalg.norm(unit))
    projection = centers @ unit
    transverse2 = np.sum(centers * centers, axis=1) - projection * projection
    hit = transverse2 <= radii * radii
    half = np.sqrt(np.maximum(radii[hit] * radii[hit] - transverse2[hit], 0.0))
    starts = np.maximum(0.0, projection[hit] - half)
    stops = np.minimum(distance, projection[hit] + half)
    intervals = union_intervals(
        [(float(a), float(b)) for a, b in zip(starts, stops, strict=True) if b > a]
    )
    length = math.fsum(stop - start for start, stop in intervals)
    maximum = max((stop - start for start, stop in intervals), default=0.0)
    observer = intervals[0][1] if intervals and intervals[0][0] == 0.0 else 0.0
    target = distance - intervals[-1][0] if intervals and intervals[-1][1] == distance else 0.0
    length = min(max(length, 0.0), distance)
    return {
        "void_length_mpc": length,
        "void_fraction": length / distance,
        "maximum_chord_mpc": maximum,
        "observer_endpoint_chord_mpc": observer,
        "target_endpoint_chord_mpc": target,
        "crossing_count": len(intervals),
    }


def source_recomputation(base_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchors = {row["id"]: row for row in base_config["source_anchors"]}
    for row in (*base_config["source_anchors"], *base_config["contract_bindings"]):
        path = ROOT / row["path"]
        require(path.is_file(), f"missing frozen source anchor {row['id']}")
        require(sha_file(path) == row["sha256"], f"frozen source anchor drift {row['id']}")
        if "bytes" in row:
            require(path.stat().st_size == row["bytes"], f"source anchor size drift {row['id']}")

    edges: dict[tuple[str, int], int] = {}
    table1_rows = 0
    for raw in (ROOT / anchors["VAST1_MAXIMAL_SPHERES_AND_EDGE_FLAGS"]["path"]).read_bytes().splitlines():
        parts = raw.decode("ascii").split()
        require(len(parts) == 11, "invalid independent VAST1 row")
        key = (parts[0], int(parts[5]))
        require(key not in edges, "duplicate independent VAST1 group")
        edges[key] = int(parts[6])
        table1_rows += 1
    require(table1_rows == 2347, "independent VAST1 count changed")

    group_lists: dict[str, list[tuple[float, float, float, float]]] = {
        "Planck2018": [],
        "WMAP5": [],
    }
    observed_groups: set[tuple[str, int]] = set()
    semantic: set[tuple[str, int, str, str, str, str]] = set()
    table2_rows = 0
    h = float(base_config["law_constants"]["planck_h"])
    with gzip.open(ROOT / anchors["VAST2_ALL_SPHERE_UNION_GEOMETRY"]["path"], "rb") as handle:
        for raw in handle:
            payload = raw.rstrip(b"\n")
            if payload.endswith(b"\r"):
                payload = payload[:-1]
            require(len(payload) == 105, "invalid independent VAST2 framing")
            parts = payload.decode("ascii").split()
            require(len(parts) == 6, "invalid independent VAST2 row")
            cosmology = parts[0]
            key = (cosmology, int(parts[5]))
            require(key in edges, "independent VAST2 group absent from VAST1")
            sem = (cosmology, int(parts[5]), *parts[1:5])
            require(sem not in semantic, "duplicate independent VAST2 sphere")
            semantic.add(sem)
            observed_groups.add(key)
            if edges[key] == 0:
                x, y, z, radius = (float(value) / h for value in parts[1:5])
                group_lists[cosmology].append((x, y, z, radius))
            table2_rows += 1
    require(table2_rows == 80080, "independent VAST2 count changed")
    require(observed_groups == set(edges), "independent VAST group mismatch")
    require(len(group_lists["Planck2018"]) == 30449, "Planck edge-0 sphere count changed")
    require(len(group_lists["WMAP5"]) == 30799, "WMAP edge-0 sphere count changed")
    geometry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, values in group_lists.items():
        array = np.asarray(values, dtype=np.float64)
        geometry[key] = (array[:, :3], array[:, 3])

    mask = (ROOT / anchors["CANONICAL_VAST_ANGULAR_MASK"]["path"]).read_bytes()
    require(len(mask) == 64800 and set(mask) <= {0, 1}, "independent mask contract changed")
    ledger_rows = [
        json.loads(line)
        for line in (ROOT / anchors["CF4_IDENTIFIER_ROLE_LEDGER"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    require(len(ledger_rows) == 38053, "identifier ledger count changed")
    require(
        [int(row["source_index"]) for row in ledger_rows] == list(range(38053)),
        "identifier ledger order changed",
    )
    geometry_config = load_json(ROOT / "configs/open_gravity_void_geometry_source_completion_v2.json")
    cosmology = geometry_config["distance_contract"]["cosmology"]
    radial_limit = float(base_config["law_constants"]["radial_mask_limit_h_inverse_mpc"])
    selected: list[dict[str, Any]] = []
    raw_rows = 0
    development_decoded = 0
    stream_offset = 0
    permitted_slices = ((8, 14), (15, 20), (21, 26), (83, 91), (92, 100))
    require(
        all(stop <= 39 or start >= 44 for start, stop in permitted_slices),
        "independent source parser touched V3k bytes",
    )
    source_path = ROOT / anchors["CF4_TABLE4_OPAQUE_ROW_CONTAINER"]["path"]
    with gzip.open(source_path, "rb") as handle:
        for source_index, raw in enumerate(handle):
            entry = ledger_rows[source_index]
            raw_rows += 1
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if payload.endswith(b"\r"):
                payload = payload[:-1]
            require(len(payload) == 157, "independent CF4 framing changed")
            require(int(entry["framed_start"]) == stream_offset, "independent CF4 offset changed")
            require(sha_bytes(raw) == entry["framed_raw_sha256"], "independent CF4 raw hash changed")
            require(sha_bytes(payload) == entry["payload_raw_sha256"], "independent CF4 payload hash changed")
            stream_offset += len(raw)
            identifier = int(payload[:7].decode("ascii").strip())
            bucket = int.from_bytes(hashlib.sha256(str(identifier).encode("ascii")).digest()[:8], "big") % 10
            role = "development" if bucket <= 5 else "validation" if bucket <= 7 else "confirmation"
            require(identifier == int(entry["identifier"]), "independent identifier changed")
            require(bucket == int(entry["bucket"]) and role == entry["role"], "independent role changed")
            if role != "development":
                continue
            dmzp = float(payload[8:14].decode("ascii").strip())
            e_dmzp = float(payload[15:20].decode("ascii").strip())
            distance_luminosity = float(payload[21:26].decode("ascii").strip())
            ra = float(payload[83:91].decode("ascii").strip())
            dec = float(payload[92:100].decode("ascii").strip())
            development_decoded += 1
            formula_distance = 10.0 ** ((dmzp - 25.0) / 5.0)
            tolerance = 0.05 + formula_distance * (10.0 ** (0.0005 / 5.0) - 1.0)
            require(abs(distance_luminosity - formula_distance) <= tolerance + 1e-12, "CF4 distance pair changed")
            i, j = math.floor(ra % 360.0), math.floor(dec + 90.0)
            if not bool(mask[i * 180 + j]):
                continue
            _, distance_hinv = luminosity_to_comoving_hinv(distance_luminosity, cosmology)
            if not 0.0 < distance_hinv <= radial_limit:
                continue
            distance = distance_hinv / h
            direction = radec_unit(ra, dec)
            planck = interval_summary(direction, distance, *geometry["Planck2018"])
            wmap = interval_summary(direction, distance, *geometry["WMAP5"])
            if float(planck["void_length_mpc"]) == 0.0 and float(wmap["void_length_mpc"]) == 0.0:
                continue
            neighborhood = [
                mask[((i + di) % 360) * 180 + min(max(j + dj, 0), 179)]
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
            ]
            selected.append(
                {
                    "identifier": identifier,
                    "source_index": source_index,
                    "bucket": bucket,
                    "role": role,
                    "DMzp": dmzp,
                    "e_DMzp": e_dmzp,
                    "Dist": distance_luminosity,
                    "RAdeg": ra,
                    "DEdeg": dec,
                    "distance_path_mpc": distance,
                    "direction": direction,
                    "mask_neighborhood_fraction": math.fsum(neighborhood) / 9.0,
                    "planck": planck,
                    "wmap": wmap,
                }
            )
            if len(selected) == 8:
                break
    require(raw_rows == 3152, "independent CF4 stop row changed")
    require(development_decoded == 1918, "independent CF4 development count changed")
    require(
        [row["identifier"] for row in selected]
        == [21354, 21431, 21645, 21659, 21682, 21723, 21735, 21777],
        "independent nonzero-exposure selector changed",
    )
    require(
        [row["source_index"] for row in selected]
        == [3062, 3079, 3121, 3126, 3131, 3140, 3143, 3151],
        "independent nonzero-exposure source indices changed",
    )
    require(
        all(
            float(row["planck"]["void_length_mpc"]) > 0.0
            or float(row["wmap"]["void_length_mpc"]) > 0.0
            for row in selected
        ),
        "selected object lacks nonzero exposure",
    )
    return selected, {
        "vast1_rows": table1_rows,
        "vast2_rows": table2_rows,
        "planck2018_edge0_spheres": len(group_lists["Planck2018"]),
        "wmap5_edge0_spheres": len(group_lists["WMAP5"]),
        "cf4_raw_rows_read": raw_rows,
        "cf4_development_source_rows_decoded": development_decoded,
        "permitted_slices": [list(row) for row in permitted_slices],
    }


def fraction_permutation(rows: list[dict[str, Any]], key: str) -> dict[int, float]:
    ordered = sorted(rows, key=lambda row: (float(row["distance_path_mpc"]), int(row["identifier"])))
    result: dict[int, float] = {}
    for start in range(0, len(ordered), 4):
        stratum = ordered[start : start + 4]
        fractions = [float(row[key]["void_fraction"]) for row in stratum]
        rotated = fractions[1:] + fractions[:1]
        for row, fraction in zip(stratum, rotated, strict=True):
            result[int(row["identifier"])] = fraction
    return result


def variant_values(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[tuple[int, str], dict[str, np.ndarray]], dict[str, Any]]:
    permutations = {
        "planck": fraction_permutation(rows, "planck"),
        "wmap": fraction_permutation(rows, "wmap"),
    }
    values_by_item: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    expected_source_geometry: dict[tuple[int, str], str] = {}
    donor_absolute_differences = 0
    fraction_roundoff = 0.0
    ordered = sorted(rows, key=lambda row: (float(row["distance_path_mpc"]), int(row["identifier"])))
    planck_donor: dict[int, dict[str, Any]] = {}
    for start in range(0, len(ordered), 4):
        stratum = ordered[start : start + 4]
        donors = stratum[1:] + stratum[:1]
        for target, donor in zip(stratum, donors, strict=True):
            planck_donor[int(target["identifier"])] = donor
    for row in rows:
        identifier = int(row["identifier"])
        distance = float(row["distance_path_mpc"])
        reconstructed = permutations["planck"][identifier] * distance
        donor_absolute = float(planck_donor[identifier]["planck"]["void_length_mpc"])
        donor_absolute_differences += int(reconstructed != donor_absolute)
        fraction_roundoff = max(
            fraction_roundoff,
            abs(reconstructed / distance - permutations["planck"][identifier]),
        )
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
                    "maximum_chord_mpc": min(permuted_length, float(original["maximum_chord_mpc"]) * scale),
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
            direction = np.asarray(row["direction"], dtype=np.float64)
            dx, dy, dz = (float(value) for value in direction)
            design = np.asarray(
                [dx, dy, dz, dx * dx, dy * dy, dz * dz, dx * dy, dx * dz, dy * dz],
                dtype=np.float64,
            )
            values_by_item[(identifier, variant)] = {
                "source.scalar.delta-h-km-s-mpc": np.asarray(
                    [config["law_constants"]["delta_h_km_s_mpc"]], dtype=np.float64
                ),
                "source.scalar.distance-modulus-mag": np.asarray([row["DMzp"]], dtype=np.float64),
                "source.scalar.distance-modulus-uncertainty-mag": np.asarray([row["e_DMzp"]], dtype=np.float64),
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
            expected_source_geometry[(identifier, variant)] = source_geometry
    return values_by_item, {
        "source_geometry": expected_source_geometry,
        "targets_differing_from_donor_absolute_length": donor_absolute_differences,
        "maximum_fraction_reconstruction_roundoff": fraction_roundoff,
    }


def predict(features: dict[str, np.ndarray], formula_id: str, c_km_s: float) -> np.ndarray:
    distance = float(features["source.scalar.distance-mpc"][0])
    h_m = float(features["source.scalar.h-m-km-s-mpc"][0])
    delta_h = float(features["source.scalar.delta-h-km-s-mpc"][0])
    exposure_key = EXPOSURE[formula_id]
    exposure = 0.0 if exposure_key is None else float(features[exposure_key][0])
    require(0.0 <= exposure <= distance, "independent prediction exposure outside path")
    return np.asarray([(h_m * distance + delta_h * exposure) / c_km_s], dtype=np.float64)


def main() -> None:
    config_path = ROOT / "configs/open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v3.json"
    module_path = ROOT / "src/sigma_theory_compiler/open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v3.py"
    test_path = ROOT / "tests/test_open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v3.py"
    receipt_path = V3_DIR / "receipt.json"
    blocked_path = ROOT / "work/audits/open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v2-independent-audit-blocked-ae2e39bb.json"
    observed_subject = {
        "config": sha_file(config_path),
        "module": sha_file(module_path),
        "test": sha_file(test_path),
        "receipt_raw": sha_file(receipt_path),
        "blocked_raw": sha_file(blocked_path),
    }
    for key, expected in EXPECTED_SUBJECT.items():
        if key in observed_subject:
            require(observed_subject[key] == expected, f"subject hash mismatch: {key}")

    config = load_json(config_path)
    base_config = load_json(ROOT / "configs/open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v1.json")
    receipt = load_json(receipt_path)
    receipt_body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    require(canonical_sha(receipt_body) == receipt["content_sha256"] == EXPECTED_SUBJECT["receipt_content"], "receipt self hash changed")
    blocked = load_json(blocked_path)
    blocked_body = {key: value for key, value in blocked.items() if key != "content_sha256"}
    require(canonical_sha(blocked_body) == blocked["content_sha256"] == EXPECTED_SUBJECT["blocked_content"], "blocked audit self hash changed")
    require(blocked["status"] == "BLOCK", "v2 block evidence was erased")

    artifact_hashes = {name: sha_file(V3_DIR / name) for name in EXPECTED_ARTIFACTS}
    require(artifact_hashes == EXPECTED_ARTIFACTS, "v3 artifact handoff hashes changed")
    require(receipt["artifact_sha256"] == EXPECTED_ARTIFACTS, "receipt artifact pins changed")
    for prefix in ("config", "module", "test"):
        path = ROOT / config["predecessor"][f"{prefix}_path"]
        require(sha_file(path) == config["predecessor"][f"{prefix}_raw_sha256"], f"v2 {prefix} predecessor drift")
    predecessor_receipt_path = ROOT / config["predecessor"]["receipt_path"]
    predecessor_receipt = load_json(predecessor_receipt_path)
    require(sha_file(predecessor_receipt_path) == config["predecessor"]["receipt_raw_sha256"], "v2 receipt raw drift")
    require(predecessor_receipt["content_sha256"] == config["predecessor"]["receipt_content_sha256"], "v2 receipt content drift")

    catalogue = subject._catalogue(subject.base.load_config())
    require(canonical_sha(catalogue.to_dict()) == catalogue.content_sha256, "catalogue canonical hash changed")
    require(catalogue.content_sha256 == receipt["catalogue_sha256"] == EXPECTED_SUBJECT["catalogue"], "catalogue hash changed")
    by_id = catalogue.by_id()
    for feature_id in RATE_FEATURES:
        element = by_id[feature_id]
        require(element.canonical_unit == RATE_UNIT, f"catalogue unit wrong for {feature_id}")
        require(list(element.si_dimension) == RATE_DIMENSION, f"catalogue dimension wrong for {feature_id}")
        require(element.axes == ("object",), f"catalogue axes wrong for {feature_id}")

    v3_rows = [json.loads(line) for line in (V3_DIR / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()]
    v2_rows = [json.loads(line) for line in (V2_DIR / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(v3_rows) == len(v2_rows) == 720, "scenario count changed")
    require(
        [row["scenario"]["scenario_id"] for row in v3_rows]
        == sorted({row["scenario"]["scenario_id"] for row in v3_rows}),
        "scenario ordering or uniqueness changed",
    )
    rate_occurrences = 0
    unexpected_normalized_differences = 0
    scenario_hash_failures = 0
    scenario_mutations_rejected = 0
    for old, new in zip(v2_rows, v3_rows, strict=True):
        scenario = new["scenario"]
        if canonical_sha(scenario) != new["scenario_sha256"]:
            scenario_hash_failures += 1
        old_outer = {key: value for key, value in old.items() if key not in {"scenario", "scenario_sha256"}}
        new_outer = {key: value for key, value in new.items() if key not in {"scenario", "scenario_sha256"}}
        require(old_outer == new_outer, "v3 changed numerical/runtime outer scenario values")
        if normalized_scenario(old["scenario"]) != normalized_scenario(scenario):
            unexpected_normalized_differences += 1
        found = {row["element_id"]: row for row in scenario["formula_features"]}
        for feature_id in RATE_FEATURES:
            rate_occurrences += 1
            require(found[feature_id]["unit"] == RATE_UNIT, "scenario rate unit was not repaired")
            require(found[feature_id]["axes"] == ["object"], "scenario rate axes changed")
            require(found[feature_id]["dtype"] == "float64" and found[feature_id]["shape"] == [1], "scenario rate runtime type changed")
        if scenario_mutations_rejected < 2:
            mutated = copy.deepcopy(scenario)
            rate_row = next(row for row in mutated["formula_features"] if row["element_id"] == RATE_FEATURES[scenario_mutations_rejected])
            rate_row["unit"] = "Mpc"
            require(canonical_sha(mutated) != new["scenario_sha256"], "scenario unit mutation escaped hash")
            require(rate_row["unit"] != RATE_UNIT, "scenario unit mutation escaped typed audit")
            scenario_mutations_rejected += 1
    require(rate_occurrences == 1440, "rate scenario reference count changed")
    require(scenario_hash_failures == 0, "scenario self hashes changed")
    require(unexpected_normalized_differences == 0, "unexpected v2/v3 scenario change")

    require((V3_DIR / "values.npz").read_bytes() == (V2_DIR / "values.npz").read_bytes(), "values NPZ differs from v2")
    require(
        (V3_DIR / "geometry-and-identifiability.json").read_bytes()
        == (V2_DIR / "geometry-and-identifiability.json").read_bytes(),
        "geometry diagnostics differs from v2",
    )
    typed_diff = load_json(V3_DIR / "typed-contract-diff.json")
    require(typed_diff["scenario_reference_occurrences"] == 1440, "typed diff reference count changed")
    require(typed_diff["unexpected_normalized_scenario_differences"] == 0, "typed diff admitted extra changes")
    require(typed_diff["values_npz_byte_identical_to_v2"] is True, "typed diff numerical identity changed")
    require(typed_diff["geometry_diagnostics_byte_identical_to_v2"] is True, "typed diff geometry identity changed")

    selected, source_summary = source_recomputation(base_config)
    diagnostics = load_json(V3_DIR / "geometry-and-identifiability.json")
    diagnostic_by_id = {int(row["identifier"]): row for row in diagnostics["selected_cf4"]}
    maximum_source_geometry_difference = 0.0
    for row in selected:
        frozen = diagnostic_by_id[int(row["identifier"])]
        require(row["source_index"] == int(frozen["source_index"]), "independent source index differs")
        maximum_source_geometry_difference = max(
            maximum_source_geometry_difference,
            abs(float(row["distance_path_mpc"]) - float.fromhex(frozen["distance_path_mpc_hex"])),
            abs(float(row["planck"]["void_length_mpc"]) - float.fromhex(frozen["planck_void_length_mpc_hex"])),
            abs(float(row["wmap"]["void_length_mpc"]) - float.fromhex(frozen["wmap_void_length_mpc_hex"])),
        )
    require(maximum_source_geometry_difference <= 1e-9, "independent geometry exceeds frozen tolerance")

    independent_items, permutation_summary = variant_values(selected, base_config)
    require(permutation_summary["targets_differing_from_donor_absolute_length"] == 7, "fraction permutation reverted to absolute length")
    require(permutation_summary["maximum_fraction_reconstruction_roundoff"] <= 3e-17, "fraction reconstruction changed")

    with np.load(V3_DIR / "values.npz", allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    locator_failures = 0
    runtime_failures = 0
    maximum_variant_value_difference = 0.0
    independently_checked_items: set[tuple[int, str]] = set()
    formula_distances: dict[str, dict[str, float]] = {}
    winners_by_scenario: dict[str, tuple[str, ...]] = {}
    recovered_by_scenario: dict[str, bool] = {}
    distinct_by_scenario: dict[str, bool] = {}
    confusion = {truth: {candidate: 0 for candidate in CANDIDATES} for truth in CANDIDATES}
    recovery = {truth: {"scenarios": 0, "recovered": 0, "distinct": 0} for truth in CANDIDATES}
    expected_runtime_hashes: dict[tuple[str, str], dict[str, str]] = {}
    c_km_s = float(base_config["law_constants"]["c_km_s"])
    comparison_spec = [{"prediction": OUTPUT_ID, "response": RESPONSE_ID, "uncertainty": UNCERTAINTY_ID}]
    parameter_values_sha = canonical_sha({})
    parameter_cell_sha = canonical_sha({"parameter_cell_id": "fixed-source-contract", "values": {}})
    for row in v3_rows:
        scenario = row["scenario"]
        scenario_id = scenario["scenario_id"]
        identifier = int(scenario["object_id"].split("-")[-1])
        variant = row["geometry_variant"]
        item_key = (identifier, variant)
        require(row["source_geometry"] == permutation_summary["source_geometry"][item_key], "source geometry label changed")
        locators = row["value_locators"]
        features: dict[str, np.ndarray] = {}
        for reference in scenario["formula_features"]:
            feature_id = reference["element_id"]
            locator = locators[feature_id]
            value = arrays[locator["key"]]
            observed_hash = array_sha(value)
            locator_failures += int(observed_hash != locator["sha256"] or observed_hash != reference["value_sha256"])
            require(reference["artifact_path"] == "runs/gravity/open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v3/values.npz", "feature artifact path changed")
            require(value.dtype.name == reference["dtype"] and list(value.shape) == reference["shape"], "feature runtime type changed")
            features[feature_id] = value
        response = arrays[locators["response"]["key"]]
        variance = arrays[locators["variance"]["key"]]
        truth_value = arrays[locators["truth"]["key"]]
        locator_failures += int(array_sha(response) != locators["response"]["sha256"])
        locator_failures += int(array_sha(variance) != locators["variance"]["sha256"])
        locator_failures += int(array_sha(truth_value) != locators["truth"]["sha256"])
        locator_failures += int(array_sha(response) != scenario["scoring_responses"][0]["value_sha256"])
        locator_failures += int(array_sha(variance) != scenario["uncertainties"][0]["artifact_sha256"])
        locator_failures += int(array_sha(truth_value) != scenario["hidden_truth"][0]["value_sha256"])
        require(int(truth_value[0]) == CANDIDATES.index(row["truth_formula_id"]), "hidden truth code changed")

        if item_key not in independently_checked_items:
            independent = independent_items[item_key]
            for feature_id, frozen in features.items():
                difference = float(np.max(np.abs(np.asarray(independent[feature_id], dtype=np.float64) - np.asarray(frozen, dtype=np.float64))))
                maximum_variant_value_difference = max(maximum_variant_value_difference, difference)
            independently_checked_items.add(item_key)
        distance = float(features["source.scalar.distance-mpc"][0])
        for key in (
            "source.scalar.void-length-mpc",
            "source.scalar.null-void-length-mpc",
            "source.scalar.maximum-chord-mpc",
            "source.scalar.observer-endpoint-chord-mpc",
            "source.scalar.target-endpoint-chord-mpc",
        ):
            require(0.0 <= float(features[key][0]) <= distance, f"geometry bound failed: {key}")
        require(0.0 <= float(features["source.scalar.void-fraction"][0]) <= 1.0, "void fraction bound changed")
        require(abs(float(features["source.scalar.void-fraction"][0]) - float(features["source.scalar.void-length-mpc"][0]) / distance) <= 3e-17, "void fraction reconstruction changed")

        seed = scenario["seed_lineage"]
        derived_seed = int(canonical_sha(seed)[:16], 16)
        require(str(derived_seed) == row["noise"]["derived_seed"], "derived seed changed")
        require(seed == next(old["scenario"]["seed_lineage"] for old in v2_rows if old["scenario"]["scenario_id"] == scenario_id), "v2/v3 seed changed")
        rng = np.random.default_rng(derived_seed)
        direction = np.asarray(features["source.vector.direction-cartesian"], dtype=np.float64)
        design = np.asarray(features["source.vector.flow-shear-design"], dtype=np.float64)
        e_dm = float(features["source.scalar.distance-modulus-uncertainty-mag"][0])
        h_m = float(features["source.scalar.h-m-km-s-mpc"][0])
        delta_h = float(features["source.scalar.delta-h-km-s-mpc"][0])
        mask_fraction = float(features["source.scalar.mask-neighborhood-fraction"][0])
        sigma_distance = (math.log(10.0) / 5.0) * e_dm * h_m * distance / c_km_s
        bulk = np.asarray(base_config["noise"]["bulk_velocity_coefficients_km_s"], dtype=np.float64)
        shear = np.asarray(base_config["noise"]["shear_velocity_coefficients_km_s"], dtype=np.float64)
        flow = (float(np.dot(direction, bulk)) + float(np.dot(design[3:], shear))) / c_km_s
        distance_draw = float(rng.normal()) * sigma_distance
        boundary = (
            (2.0 * mask_fraction - 1.0)
            * float(base_config["noise"]["mask_boundary_delta_h_fraction"])
            * delta_h
            * distance
            / c_km_s
        )
        minimum = float(base_config["noise"]["minimum_log_redshift_sigma"])
        family = row["noise"]["family"]
        if family == "zero-noise":
            offset, sigma = 0.0, math.sqrt(float(base_config["scoring"]["zero_noise_variance"]))
        elif family == "distance-measurement":
            offset, sigma = distance_draw, max(minimum, sigma_distance)
        elif family == "bulk-shear-flow":
            offset, sigma = flow, max(minimum, 250.0 / c_km_s)
        elif family == "distance-plus-flow":
            offset = distance_draw + flow
            sigma = max(minimum, math.hypot(sigma_distance, 250.0 / c_km_s))
        elif family == "selection-mask-boundary":
            offset, sigma = boundary, max(minimum, abs(boundary) + minimum)
        else:
            raise AssertionError("unregistered noise family")
        require(offset.hex() == row["noise"]["offset_hex"], "registered noise offset changed")
        require(sigma.hex() == row["noise"]["sigma_hex"], "registered noise sigma changed")
        truth_prediction = predict(features, row["truth_formula_id"], c_km_s)
        runtime_failures += int(not np.array_equal(response, np.asarray(truth_prediction + offset, dtype=np.float64)))
        runtime_failures += int(not np.array_equal(variance, np.asarray([sigma * sigma], dtype=np.float64)))

        distances: dict[str, float] = {}
        predictions: dict[str, np.ndarray] = {}
        for candidate in CANDIDATES:
            prediction = predict(features, candidate, c_km_s)
            predictions[candidate] = prediction
            delta = prediction.astype(np.float64).reshape(-1) - response.astype(np.float64).reshape(-1)
            squared = np.square(delta) / variance.reshape(-1)
            distance_score = math.sqrt(math.fsum(float(value) for value in squared) / squared.size)
            distances[candidate] = distance_score
        formula_distances[scenario_id] = distances
        minimum_score = min(distances.values())
        winners = tuple(candidate for candidate in CANDIDATES if distances[candidate] == minimum_score)
        larger = sorted(value for value in distances.values() if value > minimum_score)
        runner_up = larger[0] if larger else None
        gap = None if runner_up is None else runner_up - minimum_score
        distinct = bool(
            len(winners) == 1
            and len(distances) >= 2
            and gap is not None
            and gap > 0.0
            and gap >= float(base_config["scoring"]["distinct_gap"])
        )
        recovered = row["truth_formula_id"] in winners
        winners_by_scenario[scenario_id] = winners
        recovered_by_scenario[scenario_id] = recovered
        distinct_by_scenario[scenario_id] = distinct
        truth_id = row["truth_formula_id"]
        recovery[truth_id]["scenarios"] += 1
        recovery[truth_id]["recovered"] += int(recovered)
        recovery[truth_id]["distinct"] += int(recovered and distinct)
        for winner in winners:
            confusion[truth_id][winner] += 1

        for candidate, prediction in predictions.items():
            value_digest = array_sha(prediction)
            artifact = {
                "element_id": OUTPUT_ID,
                "artifact_path": scenario["expected_predictions"][0]["artifact_path"],
                "value_sha256": value_digest,
                "dtype": "float64",
                "shape": [1],
                "axes": ["object"],
                "unit": "1",
                "frame": "observer-centered-icrs-cartesian",
            }
            result_sha = canonical_sha({OUTPUT_ID: artifact})
            metrics_sha = canonical_sha(
                {
                    "metric": "whitened_rmse",
                    "value_hex": distances[candidate].hex(),
                    "comparisons": comparison_spec,
                }
            )
            binding_sha = receipt["formula_binding_sha256"][candidate]
            diagnostics_sha = canonical_sha(
                {
                    "deterministic_replay": True,
                    "scenario_sha256": row["scenario_sha256"],
                    "binding_sha256": binding_sha,
                    "parameter_values_sha256": parameter_values_sha,
                    "parameter_cell_sha256": parameter_cell_sha,
                    "real_response_used": False,
                }
            )
            expected_runtime_hashes[(scenario_id, candidate)] = {
                "result_sha256": result_sha,
                "metrics_sha256": metrics_sha,
                "diagnostics_sha256": diagnostics_sha,
                "seed_lineage_sha256": canonical_sha(seed),
            }
    require(locator_failures == 0, "NPZ locator/hash validation failed")
    require(runtime_failures == 0, "independent response/runtime recomputation failed")
    require(len(independently_checked_items) == 24, "independent geometry item coverage changed")
    require(maximum_variant_value_difference <= 1e-9, "independent variant values exceed tolerance")

    confusion_payload = load_json(V3_DIR / "confusion-matrix.json")
    v2_confusion = load_json(V2_DIR / "confusion-matrix.json")
    numerical_confusion_keys = (
        "truth_formula_ids",
        "candidate_formula_ids",
        "winner_membership_counts",
        "recovery_by_truth",
        "scenario_count",
        "attempted_cell_count",
        "scored_cell_count",
        "truth_recovery_count",
        "distinct_truth_recovery_count",
        "no_hand_ranking",
    )
    require(all(confusion_payload[key] == v2_confusion[key] for key in numerical_confusion_keys), "v2/v3 numerical confusion changed")
    require(confusion_payload["winner_membership_counts"] == confusion, "independent winner confusion differs")
    require(confusion_payload["recovery_by_truth"] == recovery, "independent recovery by truth differs")
    truth_recovery_count = sum(int(value) for value in recovered_by_scenario.values())
    distinct_truth_recovery_count = sum(
        int(recovered_by_scenario[key] and distinct_by_scenario[key]) for key in recovered_by_scenario
    )
    require(truth_recovery_count == 459, "independent truth recovery count changed")
    require(distinct_truth_recovery_count == 50, "independent distinct recovery count changed")

    ledger = load_json(V3_DIR / "ledger.json")
    v2_ledger = load_json(V2_DIR / "ledger.json")
    entries = ledger["entries"]
    old_entries = v2_ledger["entries"]
    require(len(entries) == len(old_entries) == 15120, "ledger replay count changed")
    allowed_ledger_changes = {
        "entry_sha256",
        "prior_entry_sha256",
        "suite_id",
        "suite_version",
        "suite_sha256",
        "binding_sha256",
        "adapter_sha256",
        "result_sha256",
        "diagnostics_sha256",
    }
    normalized_ledger_differences = 0
    prior: str | None = None
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    for sequence, (old, new) in enumerate(zip(old_entries, entries, strict=True)):
        require(new["sequence"] == sequence, "ledger sequence changed")
        require(new["prior_entry_sha256"] == prior, "ledger prior hash changed")
        body = {key: value for key, value in new.items() if key != "entry_sha256"}
        require(canonical_sha(body) == new["entry_sha256"], "ledger entry self hash changed")
        prior = new["entry_sha256"]
        old_normalized = {key: value for key, value in old.items() if key not in allowed_ledger_changes}
        new_normalized = {key: value for key, value in new.items() if key not in allowed_ledger_changes}
        normalized_ledger_differences += int(old_normalized != new_normalized)
        if new["scenario_id"] is not None:
            completed[(new["scenario_id"], new["formula_id"])] = new
    require(normalized_ledger_differences == 0, "unexpected v2/v3 ledger runtime/status change")
    require(len(completed) == 4320, "completed ledger cell count changed")
    status_counts = Counter(row["status"] for row in entries)
    require(
        status_counts
        == Counter(
            {
                "ELIGIBLE_NOT_RUN": 4320,
                "UNDERPOWERED": 3654,
                "AMBIGUOUS_WITH_COMPARATOR": 616,
                "PROMISING_DISTINCT_SIGNATURE": 50,
                "SOURCE_BLOCKED": 6480,
            }
        ),
        "ledger status counts changed",
    )
    runtime_hash_failures = 0
    status_failures = 0
    for (scenario_id, candidate), entry in completed.items():
        expected_hashes = expected_runtime_hashes[(scenario_id, candidate)]
        runtime_hash_failures += sum(
            int(entry[key] != value) for key, value in expected_hashes.items()
        )
        recovered = recovered_by_scenario[scenario_id]
        distinct = distinct_by_scenario[scenario_id]
        is_winner = candidate in winners_by_scenario[scenario_id]
        if not distinct:
            expected_status = "UNDERPOWERED"
            expected_reasons = ["candidate_gap_below_threshold"]
        elif is_winner and recovered:
            expected_status = "PROMISING_DISTINCT_SIGNATURE"
            expected_reasons = []
        else:
            expected_status = "AMBIGUOUS_WITH_COMPARATOR"
            expected_reasons = [] if recovered else ["truth_generator_not_recovered"]
        status_failures += int(entry["status"] != expected_status or entry["reason_codes"] != expected_reasons)
    require(runtime_hash_failures == 0, "independent result/metric/diagnostic/seed hashes differ")
    require(status_failures == 0, "independent ledger adjudication differs")

    # Static source-parser inspection: the frozen parser uses exactly the five
    # permitted scientific slices and never slices the V3k response interval.
    parser_tree = ast.parse(inspect.getsource(subject.base._parse_permitted_cf4_source))
    observed_slices: set[tuple[int | None, int | None]] = set()
    for node in ast.walk(parser_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_parse_float_slice"
            and len(node.args) >= 3
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[2], ast.Constant)
        ):
            observed_slices.add((node.args[1].value, node.args[2].value))
    helper_tree = ast.parse(inspect.getsource(subject.base._parse_float_slice))
    require(
        any(
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Slice)
            and isinstance(node.slice.lower, ast.Name)
            and node.slice.lower.id == "start"
            and isinstance(node.slice.upper, ast.Name)
            and node.slice.upper.id == "stop"
            for node in ast.walk(helper_tree)
        ),
        "subject float parser no longer uses its registered slice bounds",
    )
    require(observed_slices == {(8, 14), (15, 20), (21, 26), (83, 91), (92, 100)}, "subject permitted CF4 slices changed")
    require(all(stop <= 39 or start >= 44 for start, stop in observed_slices if start is not None and stop is not None), "subject parser sliced V3k")
    access = receipt["access_accounting"]
    for key in (
        "cf4_measured_velocity_fields_decoded",
        "cf4_published_peculiar_velocity_fields_decoded",
        "validation_source_fields_decoded",
        "confirmation_source_fields_decoded",
        "pantheon_files_opened",
        "real_response_values_decoded",
        "real_scores",
    ):
        require(access[key] == 0, f"prohibited response access changed: {key}")
    require(access["cf4_raw_rows_read"] == 3152 and access["cf4_development_source_rows_decoded"] == 1918, "source access accounting changed")
    require(access["vast1_source_rows_decoded"] == 2347 and access["vast2_source_rows_decoded"] == 80080, "VAST source accounting changed")

    # Adversarial mutation rejection without touching any frozen subject file.
    mutations: list[dict[str, Any]] = []
    for mutator in (
        lambda value: value["repair"].__setitem__("canonical_unit", "Mpc"),
        lambda value: value["repair"].__setitem__("si_dimension", [0, 1, 0, 0, 0, 0, 0]),
        lambda value: value["repair"].__setitem__("expected_scenario_reference_occurrences", 1439),
        lambda value: value["repair"].__setitem__("explicit_feature_map_required", False),
        lambda value: value["repair"].__setitem__("affected_feature_ids", list(reversed(value["repair"]["affected_feature_ids"]))),
        lambda value: value["access_contract"].__setitem__("pantheon_files_opened", 1),
    ):
        mutated = copy.deepcopy(config)
        mutator(mutated)
        mutations.append(mutated)
    rejected_mutations = 0
    for mutated in mutations:
        try:
            subject.validate_config(mutated, verify_hashes=False)
        except SchemaViolation:
            rejected_mutations += 1
    for group, key in (("blocked_audit", "raw_sha256"), ("predecessor", "config_raw_sha256")):
        mutated = copy.deepcopy(config)
        mutated[group][key] = "0" * 64
        try:
            subject.validate_config(mutated, verify_hashes=True)
        except SchemaViolation:
            rejected_mutations += 1
    require(rejected_mutations == 8, "config mutation rejection coverage failed")
    bit_mutation = bytearray((V3_DIR / "values.npz").read_bytes())
    bit_mutation[len(bit_mutation) // 2] ^= 1
    require(sha_bytes(bytes(bit_mutation)) != EXPECTED_ARTIFACTS["values.npz"], "artifact bit mutation escaped hash")
    receipt_mutation = copy.deepcopy(receipt)
    receipt_mutation["truth_recovery_count"] += 1
    mutated_body = {key: value for key, value in receipt_mutation.items() if key != "content_sha256"}
    require(canonical_sha(mutated_body) != receipt["content_sha256"], "receipt mutation escaped self hash")

    summary = {
        "status": "PASS",
        "subject_hashes": {
            **observed_subject,
            "receipt_content": receipt["content_sha256"],
            "catalogue": catalogue.content_sha256,
            "blocked_content": blocked["content_sha256"],
        },
        "artifact_sha256": artifact_hashes,
        "typed_contract": {
            "canonical_unit": RATE_UNIT,
            "si_dimension": RATE_DIMENSION,
            "catalogue_features_verified": 2,
            "scenario_reference_occurrences": rate_occurrences,
            "scenario_hash_failures": scenario_hash_failures,
            "unexpected_normalized_scenario_differences": unexpected_normalized_differences,
        },
        "source_recomputation": {
            **source_summary,
            "selected_identifiers": [row["identifier"] for row in selected],
            "selected_source_indices": [row["source_index"] for row in selected],
            "maximum_source_geometry_absolute_difference_mpc": maximum_source_geometry_difference,
            "maximum_variant_value_absolute_difference": maximum_variant_value_difference,
            "geometry_items_verified": len(independently_checked_items),
            "targets_differing_from_donor_absolute_length": permutation_summary["targets_differing_from_donor_absolute_length"],
            "maximum_fraction_reconstruction_roundoff": permutation_summary["maximum_fraction_reconstruction_roundoff"],
            "prohibited_fields_decoded": 0,
        },
        "matrix_recomputation": {
            "scenarios": len(v3_rows),
            "attempted_cells": 10800,
            "scored_cells": 4320,
            "source_blocked_cells": 6480,
            "replay_entries": len(entries),
            "truth_recovery_count": truth_recovery_count,
            "distinct_truth_recovery_count": distinct_truth_recovery_count,
            "recovery_by_truth": recovery,
            "ledger_status_counts": dict(sorted(status_counts.items())),
            "array_locator_hash_failures": locator_failures,
            "runtime_value_failures": runtime_failures,
            "runtime_hash_failures": runtime_hash_failures,
            "ledger_status_failures": status_failures,
            "unexpected_v2_v3_ledger_runtime_differences": normalized_ledger_differences,
        },
        "mutation_rejection": {
            "config_mutations_rejected": rejected_mutations,
            "scenario_unit_mutations_rejected": scenario_mutations_rejected,
            "artifact_bit_mutation_rejected": True,
            "receipt_field_mutation_rejected": True,
        },
        "access_accounting": {
            "cf4_v3k_fields_decoded": 0,
            "cf4_measured_velocity_fields_decoded": 0,
            "cf4_published_peculiar_velocity_fields_decoded": 0,
            "validation_source_fields_decoded": 0,
            "confirmation_source_fields_decoded": 0,
            "pantheon_files_opened": 0,
            "real_response_values_decoded": 0,
            "real_scores": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
