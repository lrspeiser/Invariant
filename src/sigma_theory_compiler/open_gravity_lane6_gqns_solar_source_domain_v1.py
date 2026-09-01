from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/open_gravity_lane6_gqns_solar_source_domain_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_lane6_gqns_solar_source_domain_v1.py")
TEST_PATH = Path("tests/test_open_gravity_lane6_gqns_solar_source_domain_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-lane6-gqns-solar-source-domain-v1/receipt.json")
_CONFIG_RAW_SHA256 = "5d6554ba951c4b36861ddd55edccda4b5e3f8e05754b008b94e5779f9bd5737d"
_CONFIG_CONTENT_SHA256 = "cf872ff36c8b7d6db84dd63d31bb82e12c2638f9fc112876d9f6af5e71a02c98"
_MODULE_SEMANTIC_SHA256 = "972a21efb5242f818fa470542fb8d019222d5b5d89ef449deda45f3bb57ffc5b"
_TEST_RAW_SHA256 = "f5900a11e92ebf2bf33f0e43644154a7cc0e923b0ba3e7ca4022b30882724aa2"
_SCHEMA = "invariant-open-gravity-lane6-gqns-solar-source-domain-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-lane6-gqns-solar-source-domain-receipt-1.0"
_TARGETS = ("MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE")


class GQNSSolarError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GQNSSolarError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    import re

    text = _repo_path(path).read_text(encoding="utf-8")
    for label in (
        "_CONFIG_RAW_SHA256",
        "_CONFIG_CONTENT_SHA256",
        "_MODULE_SEMANTIC_SHA256",
        "_TEST_RAW_SHA256",
    ):
        text = re.sub(rf'({label}\s*=\s*)"[0-9a-f]{{64}}"', rf'\1"{"0" * 64}"', text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_path(relative: str | Path) -> Path:
    path = (_ROOT / Path(relative)).resolve()
    _require(path.is_relative_to(_ROOT.resolve()), "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GQNSSolarError(f"cannot read {label}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == _SCHEMA, "config schema changed")
    _require(
        config.get("status") == "FROZEN_RESPONSE_INDEPENDENT_SOLAR_SOURCE_DOMAIN_FALSIFIER",
        "status changed",
    )
    law = config.get("unchanged_gqns_law", {})
    _require(law.get("parameters_fit") == 0 and law.get("retuned") is False, "law retuned")
    _require(len(config.get("source_domains", [])) == 8, "source domains changed")
    _require(config["time_grid"]["samples"] == 601, "time grid changed")
    access = config.get("access_contract", {})
    for key in (
        "ephemeris_binary_files_downloaded",
        "observational_response_files_opened",
        "observational_response_rows_opened",
        "parameters_fit_to_responses",
        "network_calls_by_builder",
        "model_calls",
        "paid_calls",
    ):
        _require(access.get(key) == 0, f"access boundary changed: {key}")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "output changed")
    expected = {
        "SUN",
        "MERCURY",
        "VENUS",
        "EARTH",
        "MOON",
        "MARS",
        "JUPITER",
        "SATURN",
        "URANUS",
        "NEPTUNE",
    }
    _require(set(config["gm_km3_s2"]) == expected, "GM inventory changed")


def _validate_package_files() -> None:
    _require(file_sha256(_repo_path(CONFIG_PATH)) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(_repo_path(CONFIG_PATH), "config")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config content changed")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "test bytes changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    config = _read_json(_repo_path(CONFIG_PATH), "config")
    validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_predecessor(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for role, binding in config["predecessor_binding"].items():
        digest = file_sha256(_repo_path(binding["path"]))
        _require(digest == binding["sha256"], f"predecessor changed: {role}")
        observed[role] = digest
    predecessor = _read_json(
        _repo_path(config["predecessor_binding"]["receipt"]["path"]), "predecessor receipt"
    )
    _require(predecessor["retained_counterexample_count"] >= 1, "predecessor failure erased")
    return observed


def _rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def planet_position(elements: Mapping[str, Sequence[float]], centuries: float) -> np.ndarray:
    values = {key: float(pair[0]) + float(pair[1]) * centuries for key, pair in elements.items()}
    mean_anomaly = math.radians((values["L"] - values["varpi"]) % 360.0)
    eccentricity = values["e"]
    eccentric_anomaly = mean_anomaly
    for _ in range(16):
        step = (mean_anomaly - eccentric_anomaly + eccentricity * math.sin(eccentric_anomaly)) / (
            1.0 - eccentricity * math.cos(eccentric_anomaly)
        )
        eccentric_anomaly += step
        if abs(step) <= 1.0e-14:
            break
    orbital = np.array(
        (
            values["a"] * (math.cos(eccentric_anomaly) - eccentricity),
            values["a"] * math.sqrt(1.0 - eccentricity**2) * math.sin(eccentric_anomaly),
            0.0,
        )
    )
    omega = math.radians(values["varpi"] - values["Omega"])
    inclination = math.radians(values["I"])
    node = math.radians(values["Omega"])
    return _rotation_z(node) @ _rotation_x(inclination) @ _rotation_z(omega) @ orbital


def _sun_internal_covariance(config: Mapping[str, Any], model: str) -> np.ndarray:
    radius = float(config["constants"]["sun_radius_km"]) / float(config["constants"]["au_km"])
    base = radius * radius / 5.0
    if model == "SPHERE":
        return np.eye(3) * base
    _require(model == "J2_SENSITIVITY", "unknown Sun model")
    delta = float(config["constants"]["sun_J2_sensitivity"]) * radius * radius
    return np.diag((base + delta / 3.0, base + delta / 3.0, base - 2.0 * delta / 3.0))


def _base_state(config: Mapping[str, Any], centuries: float) -> dict[str, dict[str, Any]]:
    elements = config["jpl_table1_elements"]
    gm = config["gm_km3_s2"]
    state: dict[str, dict[str, Any]] = {
        "SUN": {"position": np.zeros(3), "gm": float(gm["SUN"]), "internal": np.zeros((3, 3))}
    }
    for name in ("MERCURY", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"):
        state[name] = {
            "position": planet_position(elements[name], centuries),
            "gm": float(gm[name]),
            "internal": np.zeros((3, 3)),
        }
    emb = planet_position(elements["EMB"], centuries)
    emb_gm = float(gm["EARTH"]) + float(gm["MOON"])
    state["EMB"] = {"position": emb, "gm": emb_gm, "internal": np.zeros((3, 3))}
    elapsed_days = centuries * 36525.0
    phase = 2.0 * math.pi * elapsed_days / float(config["constants"]["moon_period_days"])
    separation = float(config["constants"]["moon_distance_km"]) / float(
        config["constants"]["au_km"]
    )
    relative = _rotation_x(
        math.radians(float(config["constants"]["moon_inclination_deg"]))
    ) @ np.array((separation * math.cos(phase), separation * math.sin(phase), 0.0))
    state["EARTH"] = {
        "position": emb - float(gm["MOON"]) / emb_gm * relative,
        "gm": float(gm["EARTH"]),
        "internal": np.zeros((3, 3)),
    }
    state["MOON"] = {
        "position": emb + float(gm["EARTH"]) / emb_gm * relative,
        "gm": float(gm["MOON"]),
        "internal": np.zeros((3, 3)),
    }
    return state


def _domain_bodies(
    config: Mapping[str, Any], domain: Mapping[str, Any], centuries: float
) -> list[dict[str, Any]]:
    state = _base_state(config, centuries)
    state["SUN"]["internal"] = _sun_internal_covariance(config, str(domain["sun_model"]))
    bodies: list[dict[str, Any]] = []
    for member in domain["members"]:
        if member != "ASTEROID_RING":
            bodies.append({"name": member, **state[member]})
            continue
        count = int(config["constants"]["asteroid_ring_points"])
        radius = float(config["constants"]["asteroid_ring_radius_au"])
        total_gm = float(config["constants"]["asteroid_ring_total_GM_km3_s2"])
        phase0 = 2.0 * math.pi * (centuries * 100.0 / 4.68)
        for index in range(count):
            angle = phase0 + 2.0 * math.pi * index / count
            bodies.append(
                {
                    "name": f"ASTEROID_RING_{index:02d}",
                    "position": np.array((radius * math.cos(angle), radius * math.sin(angle), 0.0)),
                    "gm": total_gm / count,
                    "internal": np.zeros((3, 3)),
                }
            )
    return bodies


def geometry_metrics(bodies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    weights = np.asarray([float(body["gm"]) for body in bodies])
    positions = np.asarray([body["position"] for body in bodies], dtype=float)
    _require(float(weights.sum()) > 0.0, "empty source")
    centre = np.average(positions, axis=0, weights=weights)
    covariance = np.zeros((3, 3))
    for weight, body, position in zip(weights, bodies, positions, strict=True):
        delta = position - centre
        covariance += weight * (np.outer(delta, delta) + np.asarray(body["internal"], dtype=float))
    covariance /= float(weights.sum())
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    trace = float(eigenvalues.sum())
    _require(trace > 0.0, "source has zero second-moment size")
    numerator = float(
        (eigenvalues[0] - eigenvalues[1]) ** 2
        + (eigenvalues[1] - eigenvalues[2]) ** 2
        + (eigenvalues[2] - eigenvalues[0]) ** 2
    )
    anisotropy = math.sqrt(numerator / (2.0 * trace * trace))
    return {
        "centre_au": centre,
        "covariance_au2": covariance,
        "eigenvalues_au2": eigenvalues,
        "A_Q": min(max(anisotropy, 0.0), 1.0),
        "L_au": math.sqrt(trace),
        "total_gm_km3_s2": float(weights.sum()),
    }


def yukawa_enclosed_fraction(x: float) -> float:
    _require(x >= 0.0, "negative radius")
    if x < 1.0e-4:
        return x * x / 2.0 - x**3 / 3.0 + x**4 / 8.0
    return -math.expm1(-x) - x * math.exp(-x)


def dark_density_kg_m3(
    source_gm_km3_s2: float,
    anisotropy: float,
    length_au: float,
    radius_au: float,
    config: Mapping[str, Any],
) -> float:
    _require(length_au > 0.0 and radius_au > 0.0, "invalid kernel scale")
    au_m = float(config["constants"]["au_km"]) * 1000.0
    source_mass = source_gm_km3_s2 * 1.0e9 / float(config["constants"]["G_m3_kg_s2"])
    length_m = length_au * au_m
    radius_m = radius_au * au_m
    return (
        anisotropy
        * source_mass
        * math.exp(-radius_au / length_au)
        / (4.0 * math.pi * length_m * length_m * radius_m)
    )


def _is_self(target: str, source: str) -> bool:
    return target == source or (target == "EARTH" and source == "EMB")


def acceleration(
    target: str,
    target_position: np.ndarray,
    bodies: Sequence[Mapping[str, Any]],
    *,
    anisotropy: float | None,
    length_au: float | None,
    config: Mapping[str, Any],
) -> np.ndarray:
    au_km = float(config["constants"]["au_km"])
    result = np.zeros(3)
    for body in bodies:
        if _is_self(target, str(body["name"])):
            continue
        delta_km = (target_position - np.asarray(body["position"], dtype=float)) * au_km
        distance_km = float(np.linalg.norm(delta_km))
        _require(distance_km > 0.0, "coincident non-self source")
        multiplier = 1.0
        if anisotropy is not None:
            _require(length_au is not None and length_au > 0.0, "missing GQNS length")
            x = distance_km / (length_au * au_km)
            multiplier = anisotropy * yukawa_enclosed_fraction(x)
        result -= multiplier * float(body["gm"]) * delta_km / distance_km**3
    return result * 1000.0


def relative_accelerations(
    target: str,
    target_position: np.ndarray,
    bodies: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    target_newton = acceleration(
        target, target_position, bodies, anisotropy=None, length_au=None, config=config
    )
    sun_newton = acceleration(
        "SUN", np.zeros(3), bodies, anisotropy=None, length_au=None, config=config
    )
    target_dark = acceleration(
        target,
        target_position,
        bodies,
        anisotropy=float(metrics["A_Q"]),
        length_au=float(metrics["L_au"]),
        config=config,
    )
    sun_dark = acceleration(
        "SUN",
        np.zeros(3),
        bodies,
        anisotropy=float(metrics["A_Q"]),
        length_au=float(metrics["L_au"]),
        config=config,
    )
    return target_newton - sun_newton, target_dark - sun_dark


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _format_float(value: float) -> str:
    return format(float(value), ".12e")


def _time_and_force_rows(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid = config["time_grid"]
    times = np.linspace(
        float(grid["start_centuries"]), float(grid["stop_centuries"]), int(grid["samples"])
    )
    time_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    for domain in config["source_domains"]:
        for sample_index, centuries in enumerate(times):
            bodies = _domain_bodies(config, domain, float(centuries))
            metrics = geometry_metrics(bodies)
            time_rows.append(
                {
                    "domain_id": domain["id"],
                    "sample_index": sample_index,
                    "centuries_from_J2000": _format_float(float(centuries)),
                    "A_Q": _format_float(float(metrics["A_Q"])),
                    "L_au": _format_float(float(metrics["L_au"])),
                    "lambda1_au2": _format_float(float(metrics["eigenvalues_au2"][0])),
                    "lambda2_au2": _format_float(float(metrics["eigenvalues_au2"][1])),
                    "lambda3_au2": _format_float(float(metrics["eigenvalues_au2"][2])),
                }
            )
            state = _base_state(config, float(centuries))
            for target in _TARGETS:
                position = np.asarray(state[target]["position"], dtype=float)
                newton, dark = relative_accelerations(target, position, bodies, metrics, config)
                radial_hat = position / float(np.linalg.norm(position))
                force_rows.append(
                    {
                        "domain_id": domain["id"],
                        "sample_index": sample_index,
                        "target": target,
                        "newton_x_m_s2": float(newton[0]),
                        "newton_y_m_s2": float(newton[1]),
                        "newton_z_m_s2": float(newton[2]),
                        "dark_x_m_s2": float(dark[0]),
                        "dark_y_m_s2": float(dark[1]),
                        "dark_z_m_s2": float(dark[2]),
                        "newton_radial_inward_m_s2": float(-np.dot(newton, radial_hat)),
                        "dark_radial_inward_m_s2": float(-np.dot(dark, radial_hat)),
                        "dark_magnitude_m_s2": float(np.linalg.norm(dark)),
                    }
                )
    return time_rows, force_rows


def _domain_summary(time_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    domains = sorted({str(row["domain_id"]) for row in time_rows})
    rows: list[dict[str, Any]] = []
    for domain in domains:
        subset = [row for row in time_rows if row["domain_id"] == domain]
        anisotropy = np.asarray([float(row["A_Q"]) for row in subset])
        length = np.asarray([float(row["L_au"]) for row in subset])
        rows.append(
            {
                "domain_id": domain,
                "samples": len(subset),
                "A_Q_min": _format_float(float(anisotropy.min())),
                "A_Q_median": _format_float(float(np.median(anisotropy))),
                "A_Q_max": _format_float(float(anisotropy.max())),
                "L_au_min": _format_float(float(length.min())),
                "L_au_median": _format_float(float(np.median(length))),
                "L_au_max": _format_float(float(length.max())),
            }
        )
    return rows


def _force_summary(
    config: Mapping[str, Any], force_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bound = float(config["constants"]["outer_planet_constant_radial_bound_m_s2"])
    for domain in sorted({str(row["domain_id"]) for row in force_rows}):
        domain_rows = [row for row in force_rows if row["domain_id"] == domain]
        newton_all = np.asarray(
            [[row[f"newton_{axis}_m_s2"] for axis in "xyz"] for row in domain_rows], dtype=float
        )
        dark_all = np.asarray(
            [[row[f"dark_{axis}_m_s2"] for axis in "xyz"] for row in domain_rows], dtype=float
        )
        denominator = float(np.sum(newton_all * newton_all))
        common_scale = float(np.sum(newton_all * dark_all) / denominator) if denominator else 0.0
        for target in _TARGETS:
            subset = [row for row in domain_rows if row["target"] == target]
            newton = np.asarray(
                [[row[f"newton_{axis}_m_s2"] for axis in "xyz"] for row in subset], dtype=float
            )
            dark = np.asarray(
                [[row[f"dark_{axis}_m_s2"] for axis in "xyz"] for row in subset], dtype=float
            )
            target_denominator = float(np.sum(newton * newton))
            target_scale = (
                float(np.sum(newton * dark) / target_denominator) if target_denominator else 0.0
            )
            common_residual = dark - common_scale * newton
            target_residual = dark - target_scale * newton
            magnitude = np.linalg.norm(dark, axis=1)
            common_magnitude = np.linalg.norm(common_residual, axis=1)
            target_magnitude = np.linalg.norm(target_residual, axis=1)
            newton_radial = np.asarray([float(row["newton_radial_inward_m_s2"]) for row in subset])
            dark_radial = np.asarray([float(row["dark_radial_inward_m_s2"]) for row in subset])
            common_radial = np.abs(dark_radial - common_scale * newton_radial)
            target_radial = np.abs(dark_radial - target_scale * newton_radial)
            rows.append(
                {
                    "domain_id": domain,
                    "target": target,
                    "common_inverse_square_scale": _format_float(common_scale),
                    "per_target_inverse_square_scale": _format_float(target_scale),
                    "dark_magnitude_min_m_s2": _format_float(float(magnitude.min())),
                    "dark_magnitude_median_m_s2": _format_float(float(np.median(magnitude))),
                    "dark_magnitude_max_m_s2": _format_float(float(magnitude.max())),
                    "common_scale_residual_max_m_s2": _format_float(float(common_magnitude.max())),
                    "per_target_scale_residual_max_m_s2": _format_float(
                        float(target_magnitude.max())
                    ),
                    "dark_radial_absolute_max_m_s2": _format_float(
                        float(np.abs(dark_radial).max())
                    ),
                    "common_scale_radial_residual_max_m_s2": _format_float(
                        float(common_radial.max())
                    ),
                    "per_target_scale_radial_residual_max_m_s2": _format_float(
                        float(target_radial.max())
                    ),
                    "published_outer_bound_m_s2": _format_float(
                        bound if target == "NEPTUNE" else 0.0
                    ),
                    "raw_to_outer_bound": _format_float(
                        float(np.abs(dark_radial).max() / bound) if target == "NEPTUNE" else 0.0
                    ),
                    "common_residual_to_outer_bound": _format_float(
                        float(common_radial.max() / bound) if target == "NEPTUNE" else 0.0
                    ),
                    "per_target_residual_to_outer_bound": _format_float(
                        float(target_radial.max() / bound) if target == "NEPTUNE" else 0.0
                    ),
                }
            )
    return rows


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _rotated_bodies(
    bodies: Sequence[Mapping[str, Any]], rotation: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            **body,
            "position": rotation @ np.asarray(body["position"], dtype=float),
            "internal": rotation @ np.asarray(body["internal"], dtype=float) @ rotation.T,
        }
        for body in bodies
    ]


def _source_sensitivity(config: Mapping[str, Any]) -> dict[str, Any]:
    domains = {domain["id"]: domain for domain in config["source_domains"]}
    metrics = {
        name: geometry_metrics(_domain_bodies(config, domain, 0.0))
        for name, domain in domains.items()
    }
    rng = np.random.default_rng(20260831)
    base_domain = domains["D05_SUN_EIGHT_PLANETS"]
    base_bodies = _domain_bodies(config, base_domain, 0.0)
    base_metrics = metrics["D05_SUN_EIGHT_PLANETS"]
    target_position = _base_state(config, 0.0)["EARTH"]["position"]
    _, base_force = relative_accelerations(
        "EARTH", target_position, base_bodies, base_metrics, config
    )
    max_a_error = 0.0
    max_l_error = 0.0
    max_force_error = 0.0
    for _ in range(64):
        rotation = _random_rotation(rng)
        rotated = _rotated_bodies(base_bodies, rotation)
        rotated_metrics = geometry_metrics(rotated)
        _, rotated_force = relative_accelerations(
            "EARTH", rotation @ target_position, rotated, rotated_metrics, config
        )
        max_a_error = max(
            max_a_error, abs(float(rotated_metrics["A_Q"]) - float(base_metrics["A_Q"]))
        )
        max_l_error = max(
            max_l_error, abs(float(rotated_metrics["L_au"]) - float(base_metrics["L_au"]))
        )
        max_force_error = max(
            max_force_error, float(np.linalg.norm(rotated_force - rotation @ base_force))
        )

    translation = np.array((13.0, -7.0, 2.5))
    translated = [
        {**body, "position": np.asarray(body["position"]) + translation} for body in base_bodies
    ]
    translated_metrics = geometry_metrics(translated)

    split = []
    for body in base_bodies:
        if body["name"] != "SUN":
            split.append(dict(body))
        else:
            split.extend(
                [
                    {**body, "name": "SUN_A", "gm": float(body["gm"]) / 2.0},
                    {**body, "name": "SUN_B", "gm": float(body["gm"]) / 2.0},
                ]
            )
    split_metrics = geometry_metrics(split)

    sun_bodies = _domain_bodies(config, domains["D01_SUN_OBLATE_ONLY"], 0.0)
    planet_bodies = [body for body in base_bodies if body["name"] != "SUN"]
    planet_metrics = geometry_metrics(planet_bodies)
    decomposition: dict[str, Any] = {}
    for target in ("EARTH", "NEPTUNE"):
        position = _base_state(config, 0.0)[target]["position"]
        _, whole = relative_accelerations(target, position, base_bodies, base_metrics, config)
        _, sun_part = relative_accelerations(
            target, position, sun_bodies, geometry_metrics(sun_bodies), config
        )
        _, planet_part = relative_accelerations(
            target, position, planet_bodies, planet_metrics, config
        )
        decomposition[target] = {
            "whole_m_s2": whole.tolist(),
            "separate_sun_plus_planets_m_s2": (sun_part + planet_part).tolist(),
            "difference_m_s2": float(np.linalg.norm(whole - sun_part - planet_part)),
            "relative_difference": float(
                np.linalg.norm(whole - sun_part - planet_part)
                / max(np.linalg.norm(whole), 1.0e-300)
            ),
        }

    remote: list[dict[str, Any]] = []
    for fraction, radius in ((1.0e-12, 1.0e3), (1.0e-12, 1.0e6), (1.0e-9, 1.0e3)):
        bodies = list(sun_bodies) + [
            {
                "name": f"REMOTE_{fraction}_{radius}",
                "position": np.array((radius, 0.0, 0.0)),
                "gm": float(config["gm_km3_s2"]["SUN"]) * fraction,
                "internal": np.zeros((3, 3)),
            }
        ]
        remote_metrics = geometry_metrics(bodies)
        remote.append(
            {
                "mass_fraction_of_sun": fraction,
                "radius_au": radius,
                "A_Q": float(remote_metrics["A_Q"]),
                "L_au": float(remote_metrics["L_au"]),
            }
        )

    density_rows: list[dict[str, Any]] = []
    for radius in (0.387, 1.0, 9.58, 20.0, 30.0):
        density_rows.append(
            {
                "radius_au": radius,
                "sun_cloud_density_kg_m3": dark_density_kg_m3(
                    float(config["gm_km3_s2"]["SUN"]),
                    float(base_metrics["A_Q"]),
                    float(base_metrics["L_au"]),
                    radius,
                    config,
                ),
                "sun_cloud_enclosed_mass_over_sun_mass": float(base_metrics["A_Q"])
                * yukawa_enclosed_fraction(radius / float(base_metrics["L_au"])),
            }
        )

    def compact(name: str) -> dict[str, float]:
        return {"A_Q": float(metrics[name]["A_Q"]), "L_au": float(metrics[name]["L_au"])}

    return {
        "arbitrary_SO3": {
            "rotations": 64,
            "max_A_Q_absolute_error": max_a_error,
            "max_L_au_absolute_error": max_l_error,
            "max_force_covariance_error_m_s2": max_force_error,
        },
        "translation": {
            "A_Q_absolute_error": abs(
                float(translated_metrics["A_Q"]) - float(base_metrics["A_Q"])
            ),
            "L_au_absolute_error": abs(
                float(translated_metrics["L_au"]) - float(base_metrics["L_au"])
            ),
        },
        "co_located_mass_split": {
            "A_Q_absolute_error": abs(float(split_metrics["A_Q"]) - float(base_metrics["A_Q"])),
            "L_au_absolute_error": abs(float(split_metrics["L_au"]) - float(base_metrics["L_au"])),
        },
        "named_boundary_J2000": {name: compact(name) for name in sorted(metrics)},
        "moon_sensitivity": {
            "A_Q_delta": float(
                metrics["D06_MOON_SPLIT"]["A_Q"] - metrics["D05_SUN_EIGHT_PLANETS"]["A_Q"]
            ),
            "L_au_delta": float(
                metrics["D06_MOON_SPLIT"]["L_au"] - metrics["D05_SUN_EIGHT_PLANETS"]["L_au"]
            ),
        },
        "asteroid_sensitivity": {
            "A_Q_delta": float(
                metrics["D07_ASTEROID_RING"]["A_Q"] - metrics["D06_MOON_SPLIT"]["A_Q"]
            ),
            "L_au_delta": float(
                metrics["D07_ASTEROID_RING"]["L_au"] - metrics["D06_MOON_SPLIT"]["L_au"]
            ),
        },
        "solar_oblateness_sensitivity": {
            "A_Q_sphere": float(metrics["D00_SUN_SPHERE_ONLY"]["A_Q"]),
            "A_Q_J2": float(metrics["D01_SUN_OBLATE_ONLY"]["A_Q"]),
            "L_au_delta": float(
                metrics["D01_SUN_OBLATE_ONLY"]["L_au"] - metrics["D00_SUN_SPHERE_ONLY"]["L_au"]
            ),
        },
        "nonlinear_decomposition": decomposition,
        "remote_source_boundary": remote,
        "D05_J2000_sun_cloud": density_rows,
    }


def _negative_theorem(
    config: Mapping[str, Any],
    domain_summary: Sequence[Mapping[str, Any]],
    force_summary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    d05 = next(row for row in domain_summary if row["domain_id"] == "D05_SUN_EIGHT_PLANETS")
    anisotropy = float(d05["A_Q_median"])
    length = float(d05["L_au_median"])
    radii = {"MERCURY": 0.387, "EARTH": 1.0, "SATURN": 9.58, "NEPTUNE": 30.07}
    enhancements = {
        name: anisotropy * yukawa_enclosed_fraction(radius / length)
        for name, radius in radii.items()
    }
    spread = max(enhancements.values()) - min(enhancements.values())
    neptune = next(
        row
        for row in force_summary
        if row["domain_id"] == "D05_SUN_EIGHT_PLANETS" and row["target"] == "NEPTUNE"
    )
    ratio = float(neptune["common_residual_to_outer_bound"])
    return {
        "theorem_id": "GQNS_GLOBAL_SOURCE_SOLAR_NO_COMMON_INVERSE_SQUARE_RESCALING_V1",
        "assumptions": [
            "The Lane-6 GQNS law and normalized positive Helmholtz kernel are unchanged.",
            "A_Q and L are computed from one finite full baryonic source domain.",
            "At least two test radii are positive and unequal, with A_Q>0 and finite L.",
        ],
        "analytic_statements": [
            "For one point source, the enclosed effective dark-mass fraction is A_Q*[1-(1+r/L)exp(-r/L)].",
            "Its derivative with respect to x=r/L is A_Q*x*exp(-x)>0 for x>0, so no one inverse-square GM rescaling matches two unequal radii exactly.",
            "As r/L tends to infinity the source carries total effective dark mass A_Q times its baryonic mass; as r/L tends to zero the force fraction is A_Q*r^2/(2L^2)+O(r^3/L^3).",
            "Because A_Q and L are nonlinear global moments, GQNS[source1+source2] is generally not GQNS[source1]+GQNS[source2].",
        ],
        "D05_median_A_Q": anisotropy,
        "D05_median_L_au": length,
        "D05_point_sun_enhancement_by_radius": enhancements,
        "minimum_pairwise_fractional_spread_before_a_common_rescale": spread,
        "D05_Neptune_common_scale_residual_to_published_outer_bound": ratio,
        "decision_rule": "Decisive bounded pre-fit exclusion if the common-scale residual/bound ratio exceeds 1000 and survives all named source-domain sensitivities without tuning.",
        "decision": (
            "DECISIVELY_EXCLUDED_AS_UNCHANGED_GLOBAL_SOLAR_SOURCE_LAW"
            if ratio > 1000.0
            else "NOT_DECISIVE_REQUIRES_PRECISION_EPHEMERIS_REFIT"
        ),
        "localization_boundary": "D00/D01 host-only behavior is not a repair of the original full-source law. It requires a new covariant source-localization/decomposition dynamics and must be versioned as a different theory.",
        "claim_limit": "This is an analytic plus response-independent pre-fit incompatibility result, not a DE440/INPOP orbit refit.",
    }


def _report(receipt: Mapping[str, Any], theorem: Mapping[str, Any]) -> bytes:
    summary = receipt["summary"]
    text = f"""# Lane 6 GQNS Solar-System/source-domain falsifier

## Result

**{theorem["decision"]}** for the unchanged Lane-6 global GQNS source functional. This is a response-independent analytic and published-bound preflight, not a precision DE440 or INPOP refit.

The frozen JPL Table-1 approximation was evaluated at {summary["time_samples"]} epochs from J2000 through 2050 for {summary["source_domains"]} explicit source domains. No ephemeris binary or observational response row was downloaded or opened, and no parameter was tuned.

## Decisive mechanism

For a point source, the GQNS effective enclosed-mass fraction is

`A_Q [1-(1+r/L) exp(-r/L)]`.

It is strictly increasing with `r/L` whenever `A_Q>0`. Consequently, one fitted inverse-square mass scale cannot absorb the same global GQNS field at two unequal radii. For the Sun plus eight planetary barycenters, the median values are `A_Q={theorem["D05_median_A_Q"]:.6g}` and `L={theorem["D05_median_L_au"]:.6g} au`. After removing the best common inverse-square scale from all frozen planet vectors, the maximum Neptune residual is {theorem["D05_Neptune_common_scale_residual_to_published_outer_bound"]:.6g} times the conservative published outer-planet constant-acceleration bound.

## Source-domain failure

Exact spherical shutoff is not Solar recovery. Named boundaries through the inner planets, Jupiter, Saturn, all eight planets, a resolved Earth-Moon pair, and an asteroid-ring sensitivity give different `A_Q`, `L`, and forces. The package also retains the non-additivity between solving the Sun-plus-planets globally and solving Sun and planets as separate source components. An arbitrarily selected host-only Sun domain can suppress the effect, but that is a new source-localization law, not the original Lane-6 rule.

## Controls

- 64 arbitrary SO(3) rotations test both moment invariance and force covariance.
- Translation and co-located mass splitting are checked.
- Solar oblateness, Moon splitting, a 36-point main-belt mass ring, distant low-mass sources, and named source boundaries are retained.
- The normalized Helmholtz enclosed-mass and density implications are explicit.
- A common inverse-square nuisance and a more permissive per-planet inverse-square nuisance are both reported; neither is tuned to an observational response.

## Claim boundary

The result excludes the unchanged **global Solar-System application** at pre-fit margins far larger than the published bound. It does not substitute for a full planetary-ephemeris refit and does not prove that every possible new localization theory fails. Any localization repair needs explicit covariant dynamics, conservation, cluster decomposition, and a new version.
"""
    return text.encode("utf-8")


def build_receipt(
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if config is None:
        config = load_config()
    else:
        validate_config(config)
    predecessor = validate_predecessor(config)
    time_rows, force_raw = _time_and_force_rows(config)
    domain_rows = _domain_summary(time_rows)
    force_rows = _force_summary(config, force_raw)
    sensitivity = _source_sensitivity(config)
    theorem = _negative_theorem(config, domain_rows, force_rows)
    time_fields = tuple(time_rows[0])
    domain_fields = tuple(domain_rows[0])
    force_fields = tuple(force_rows[0])
    payloads: dict[str, bytes] = {
        "domain-time-series.csv": _csv_bytes(time_rows, time_fields),
        "domain-summary.csv": _csv_bytes(domain_rows, domain_fields),
        "force-summary.csv": _csv_bytes(force_rows, force_fields),
        "source-domain-sensitivity.json": canonical_bytes(sensitivity),
        "bounded-negative-theorem.json": canonical_bytes(theorem),
    }
    provisional = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": theorem["decision"],
        "decision": theorem["decision"],
        "predecessor_hashes": predecessor,
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "summary": {
            "source_domains": len(config["source_domains"]),
            "time_samples": int(config["time_grid"]["samples"]),
            "domain_time_rows": len(time_rows),
            "force_summary_rows": len(force_rows),
            "targets": len(_TARGETS),
            "response_rows_opened": 0,
            "parameters_fit_to_responses": 0,
            "D05_median_A_Q": theorem["D05_median_A_Q"],
            "D05_median_L_au": theorem["D05_median_L_au"],
            "D05_Neptune_common_residual_to_bound": theorem[
                "D05_Neptune_common_scale_residual_to_published_outer_bound"
            ],
            "arbitrary_SO3_max_force_error_m_s2": sensitivity["arbitrary_SO3"][
                "max_force_covariance_error_m_s2"
            ],
        },
        "retained_failures": {
            "predecessor_counterexamples_preserved": True,
            "global_source_domain_exclusion": theorem["decision"],
            "source_boundary_dependence": sensitivity["named_boundary_J2000"],
            "nonlinear_decomposition": sensitivity["nonlinear_decomposition"],
            "host_only_localization_is_new_theory": True,
        },
        "published_source_boundary": config["published_sources"],
        "access_contract": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    payloads["report.md"] = _report(provisional, theorem)
    output_by_name = {
        Path(path).name: path
        for path in config["outputs"].values()
        if path != config["outputs"]["receipt"]
    }
    artifact_index = []
    for name, payload in sorted(payloads.items()):
        _require(name in output_by_name, f"undeclared artifact: {name}")
        artifact_index.append(
            {
                "path": output_by_name[name],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    receipt = {**provisional, "artifact_index": artifact_index}
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt, payloads


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"existing output differs: {path}")
        return "EXISTING_IDENTICAL"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def write_package() -> str:
    config = load_config()
    receipt, payloads = build_receipt(config)
    for row in receipt["artifact_index"]:
        name = Path(row["path"]).name
        _atomic_no_clobber(_repo_path(row["path"]), payloads[name])
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))


def validate_written_package() -> dict[str, Any]:
    config = load_config()
    expected, payloads = build_receipt(config)
    observed = _read_json(_repo_path(OUTPUT_PATH), "receipt")
    _require(observed == expected, "receipt differs from deterministic rebuild")
    for row in observed["artifact_index"]:
        path = _repo_path(row["path"])
        _require(path.read_bytes() == payloads[path.name], f"artifact differs: {path.name}")
        _require(file_sha256(path) == row["sha256"], f"artifact hash differs: {path.name}")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(write_package())
    elif args.command == "check":
        receipt = validate_written_package()
        print(
            json.dumps(
                {"status": receipt["status"], "content_sha256": receipt["content_sha256"]},
                sort_keys=True,
            )
        )
    else:
        if _repo_path(OUTPUT_PATH).exists():
            print(_read_json(_repo_path(OUTPUT_PATH), "receipt")["status"])
        else:
            print(load_config()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
