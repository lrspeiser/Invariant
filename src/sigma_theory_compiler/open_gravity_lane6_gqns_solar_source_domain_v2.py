"""Append-only corrected Lane-6 GQNS Solar source-domain stress test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_lane6_gqns_solar_source_domain_v1 as lane1

_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/open_gravity_lane6_gqns_solar_source_domain_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_lane6_gqns_solar_source_domain_v2.py")
TEST_PATH = Path("tests/test_open_gravity_lane6_gqns_solar_source_domain_v2.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-lane6-gqns-solar-source-domain-v2/receipt.json")
_CONFIG_RAW_SHA256 = "eb4959abc1bf790b8a595bf523f66c14df3acbfef5cf0fb9ee5f39ea248d249a"
_CONFIG_CONTENT_SHA256 = "5cf5d84a05c7781dd7f61eff9224242605118ebc66fcbd26d8c5dfe0ad494759"
_MODULE_SEMANTIC_SHA256 = "0b8ef34dc18c6e10205e0e5014cc60c1126a1f4339823ce3f66e1ea005916470"
_TEST_RAW_SHA256 = "6b903fa8bc814f0e05847583fa2eab250e0e5943584b3f835c5f53dee5f82d17"
_SCHEMA = "invariant-open-gravity-lane6-gqns-solar-source-domain-2.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-lane6-gqns-solar-source-domain-receipt-2.0"
_PACKAGE_ID = "open-gravity-lane6-gqns-solar-source-domain-v2"
_STATUS = "PASS_SOURCE_ONLY_STRESS_TEST__OBSERVATIONAL_EXCLUSION_BLOCKED"


class GQNSSolarV2Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GQNSSolarV2Error(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def content_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(relative: str | Path) -> Path:
    path = (_ROOT / Path(relative)).resolve()
    _require(path.is_relative_to(_ROOT.resolve()), "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GQNSSolarV2Error(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = _repo_path(path).read_text(encoding="utf-8")
    for label in (
        "_CONFIG_RAW_SHA256",
        "_CONFIG_CONTENT_SHA256",
        "_MODULE_SEMANTIC_SHA256",
        "_TEST_RAW_SHA256",
    ):
        text, count = re.subn(
            rf'({label}\s*=\s*)"[0-9a-f]{{64}}"', rf'\1"{"0" * 64}"', text
        )
        _require(count == 1, f"semantic pin count changed: {label}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == _SCHEMA, "config schema changed")
    _require(config.get("package_id") == _PACKAGE_ID, "package id changed")
    _require(config.get("status") == _STATUS, "status changed")
    _require(config.get("supersedes", {}).get("package_id") == lane1.load_config()["package_id"], "predecessor changed")
    corrections = config.get("corrections", {})
    _require(
        corrections.get("replacement_observational_decision")
        == "NOT_EVALUATED__MATCHED_N_BODY_REFIT_REQUIRED",
        "observational decision changed",
    )
    _require(
        corrections.get("invalid_ratio_retained_only_as_failure") == 21632.16367487,
        "withdrawn ratio history changed",
    )
    preflight = config.get("matched_ephemeris_refit_preflight", {})
    _require(
        preflight.get("status")
        == "FROZEN_SOURCE_AND_SOLVER_CONTRACT__EXECUTION_BLOCKED_BEFORE_RESPONSE_ACCESS",
        "preflight status changed",
    )
    reference = preflight.get("reference_source", {})
    _require(reference.get("filename") == "de440.bsp", "reference ephemeris changed")
    _require(reference.get("published_md5") == "c9d581bfd84209dbeee8b1583939b148", "DE440 checksum changed")
    _require(reference.get("sha256") is None, "unopened DE440 SHA-256 invented")
    _require(reference.get("payload_opened") is False, "DE440 payload state changed")
    solver = preflight.get("solver", {})
    _require(solver.get("integrator") == "scipy.integrate.solve_ivp DOP853", "integrator changed")
    _require(solver.get("implementation_complete") is False, "solver completion invented")
    _require(solver.get("implementation_sha256") is None, "solver hash invented")
    fit = preflight.get("fit_contract", {})
    _require(fit.get("same_response_rows_and_weights_for_every_model") is True, "matched-fit rule changed")
    _require(fit.get("decision_authority") is False, "premature decision authority")
    response = preflight.get("response_gate", {})
    for key in ("observational_files_opened", "observational_rows_opened", "residual_values_opened"):
        _require(response.get(key) == 0, f"response access changed: {key}")
    _require(response.get("opening_authorized") is False, "response opening authorized")
    access = config.get("access_contract", {})
    for key in (
        "builder_network_calls",
        "ephemeris_binary_files_opened",
        "observational_response_files_opened",
        "observational_response_rows_opened",
        "residual_values_opened",
        "parameters_fit_to_responses",
    ):
        _require(access.get(key) == 0, f"access boundary changed: {key}")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "output changed")


def _validate_package_files() -> dict[str, str]:
    observed = {
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(_read_json(_repo_path(CONFIG_PATH), "config")),
        "module_semantic_sha256": module_semantic_sha256(),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
    }
    expected = {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }
    _require(observed == expected, "package files changed")
    return observed


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    config = _read_json(_repo_path(CONFIG_PATH), "config")
    validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_predecessor(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    predecessor = config["supersedes"]
    for role in ("config", "module", "test", "receipt"):
        binding = predecessor[role]
        digest = file_sha256(_repo_path(binding["path"]))
        _require(digest == binding["sha256"], f"v1 predecessor changed: {role}")
        observed[role] = digest
    v1_receipt = lane1.validate_written_package()
    _require(
        v1_receipt["content_sha256"] == predecessor["receipt_content_sha256"],
        "v1 receipt content changed",
    )
    _require(
        v1_receipt["decision"] == "DECISIVELY_EXCLUDED_AS_UNCHANGED_GLOBAL_SOLAR_SOURCE_LAW",
        "v1 withdrawn decision history changed",
    )
    return observed


@lru_cache(maxsize=1)
def _v1_rebuild() -> tuple[dict[str, Any], dict[str, bytes]]:
    return lane1.build_receipt()


def _csv_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def _enclosed_fraction_independent(x: float) -> float:
    _require(x >= 0.0, "negative independent kernel radius")
    if x < 1.0e-5:
        return x * x / 2.0 - x**3 / 3.0 + x**4 / 8.0
    return 1.0 - (1.0 + x) * math.exp(-x)


def _manual_acceleration(
    target: str,
    target_position: np.ndarray,
    bodies: Sequence[Mapping[str, Any]],
    *,
    anisotropy: float | None,
    length_au: float | None,
    config: Mapping[str, Any],
) -> np.ndarray:
    au_km = float(config["constants"]["au_km"])
    total = np.zeros(3)
    for body in bodies:
        source = str(body["name"])
        if target == source or (target == "EARTH" and source == "EMB"):
            continue
        delta_km = (target_position - np.asarray(body["position"], dtype=float)) * au_km
        distance_km = float(np.linalg.norm(delta_km))
        _require(distance_km > 0.0, "manual non-self coincidence")
        factor = 1.0
        if anisotropy is not None:
            _require(length_au is not None and length_au > 0.0, "manual length missing")
            factor = anisotropy * _enclosed_fraction_independent(
                distance_km / (length_au * au_km)
            )
        total -= factor * float(body["gm"]) * delta_km / distance_km**3
    return total * 1000.0


def _projection_scale(template: np.ndarray, signal: np.ndarray) -> float:
    denominator = float(np.vdot(template, template))
    _require(denominator > 0.0, "empty projection template")
    return float(np.vdot(template, signal) / denominator)


def _normal_equation_error(template: np.ndarray, residual: np.ndarray) -> float:
    numerator = abs(float(np.vdot(template, residual)))
    denominator = math.sqrt(float(np.vdot(template, template)) * float(np.vdot(residual, residual)))
    return numerator / max(denominator, 1.0e-300)


def _corrected_theorem(config: Mapping[str, Any]) -> dict[str, Any]:
    _, payloads = _v1_rebuild()
    domains = _csv_rows(payloads["domain-summary.csv"])
    force = _csv_rows(payloads["force-summary.csv"])
    d05 = next(row for row in domains if row["domain_id"] == "D05_SUN_EIGHT_PLANETS")
    anisotropy = float(d05["A_Q_median"])
    length = float(d05["L_au_median"])
    radii = {"MERCURY": 0.387, "EARTH": 1.0, "SATURN": 9.58, "NEPTUNE": 30.07}
    enhancements = {
        name: anisotropy * _enclosed_fraction_independent(radius / length)
        for name, radius in radii.items()
    }
    pairwise = []
    names = sorted(enhancements)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "absolute_spread": abs(enhancements[left] - enhancements[right]),
                }
            )
    spreads = [row["absolute_spread"] for row in pairwise]
    maximum = max(enhancements.values()) - min(enhancements.values())
    _require(abs(maximum - max(spreads)) <= 1.0e-15, "pairwise maximum mismatch")
    neptune = next(
        row
        for row in force
        if row["domain_id"] == "D05_SUN_EIGHT_PLANETS" and row["target"] == "NEPTUNE"
    )
    sensitivity = json.loads(payloads["source-domain-sensitivity.json"])
    named = sensitivity["named_boundary_J2000"]
    full = {name: named[name] for name in ("D05_SUN_EIGHT_PLANETS", "D06_MOON_SPLIT", "D07_ASTEROID_RING")}
    return {
        "theorem_id": "GQNS_GLOBAL_SOURCE_SOLAR_SOURCE_ONLY_STRESS_TEST_V2",
        "status": _STATUS,
        "observational_decision": "NOT_EVALUATED__MATCHED_N_BODY_REFIT_REQUIRED",
        "analytic_result": {
            "point_source_enclosed_fraction": "A_Q*[1-(1+r/L)exp(-r/L)]",
            "derivative_in_x": "A_Q*x*exp(-x)>0 for A_Q>0 and x>0",
            "conclusion": "For fixed positive A_Q and finite L, no exact common inverse-square constant matches two unequal positive radii.",
        },
        "D05_median_A_Q": anisotropy,
        "D05_median_L_au": length,
        "D05_point_sun_enhancement_by_radius": enhancements,
        "pairwise_absolute_spreads": pairwise,
        "minimum_pairwise_absolute_spread": min(spreads),
        "maximum_pairwise_absolute_spread": max(spreads),
        "minimax_common_constant_maximum_residual_fraction": maximum / 2.0,
        "D05_Neptune_source_only_acceleration": {
            "dark_radial_absolute_max_m_s2": float(neptune["dark_radial_absolute_max_m_s2"]),
            "common_projection_radial_residual_max_m_s2": float(
                neptune["common_scale_radial_residual_max_m_s2"]
            ),
            "per_target_projection_radial_residual_max_m_s2": float(
                neptune["per_target_scale_radial_residual_max_m_s2"]
            ),
            "interpretation": "Source-only frozen-trajectory stress amplitudes; not observational bounds or postfit residuals.",
        },
        "domain_statement": {
            "D05_D06_D07_declared_refinement_values": full,
            "host_only_failures": {name: named[name] for name in ("D00_SUN_SPHERE_ONLY", "D01_SUN_OBLATE_ONLY")},
            "other_named_boundary_failures": {name: named[name] for name in ("D02_SUN_INNER_BARYCENTERS", "D03_SUN_THROUGH_JUPITER", "D04_SUN_THROUGH_SATURN")},
            "remote_source_boundary": sensitivity["remote_source_boundary"],
            "claim": "Only the D05-D06-D07 Moon/asteroid refinements are called stable; all other changes remain localization failures.",
        },
        "superseded_failure": {
            "v1_ratio": config["corrections"]["invalid_ratio_retained_only_as_failure"],
            "v1_decision": config["corrections"]["withdrawn_observational_decision"],
            "reason": config["corrections"]["invalid_ratio_reason"],
            "used_for_v2_decision": False,
        },
        "claim_limit": "Analytic and source-only stress test only. No DE440/INPOP observational exclusion or postfit result.",
    }


def _independent_checks() -> dict[str, Any]:
    config = lane1.load_config()
    domains = {row["id"]: row for row in config["source_domains"]}
    self_norms: dict[str, float] = {}
    state = lane1._base_state(config, 0.0)
    for target in lane1._TARGETS:
        source = "EMB" if target == "EARTH" else target
        body = {"name": source, **state[source]}
        measured = lane1.acceleration(
            target,
            np.asarray(state[target]["position"], dtype=float),
            [body],
            anisotropy=0.7,
            length_au=0.4,
            config=config,
        )
        self_norms[f"{target}_AS_{source}"] = float(np.linalg.norm(measured))

    max_newton_error = 0.0
    max_dark_error = 0.0
    cases = 0
    for domain_id in ("D05_SUN_EIGHT_PLANETS", "D06_MOON_SPLIT"):
        for centuries in (0.0, 0.137, 0.5):
            bodies = lane1._domain_bodies(config, domains[domain_id], centuries)
            metrics = lane1.geometry_metrics(bodies)
            epoch_state = lane1._base_state(config, centuries)
            for target in ("EARTH", "NEPTUNE"):
                position = np.asarray(epoch_state[target]["position"], dtype=float)
                observed_newton, observed_dark = lane1.relative_accelerations(
                    target, position, bodies, metrics, config
                )
                manual_newton = _manual_acceleration(
                    target, position, bodies, anisotropy=None, length_au=None, config=config
                ) - _manual_acceleration(
                    "SUN", np.zeros(3), bodies, anisotropy=None, length_au=None, config=config
                )
                manual_dark = _manual_acceleration(
                    target,
                    position,
                    bodies,
                    anisotropy=float(metrics["A_Q"]),
                    length_au=float(metrics["L_au"]),
                    config=config,
                ) - _manual_acceleration(
                    "SUN",
                    np.zeros(3),
                    bodies,
                    anisotropy=float(metrics["A_Q"]),
                    length_au=float(metrics["L_au"]),
                    config=config,
                )
                max_newton_error = max(
                    max_newton_error, float(np.max(np.abs(observed_newton - manual_newton)))
                )
                max_dark_error = max(
                    max_dark_error, float(np.max(np.abs(observed_dark - manual_dark)))
                )
                cases += 1

    oblate = lane1._domain_bodies(config, domains["D01_SUN_OBLATE_ONLY"], 0.0)
    metrics = lane1.geometry_metrics(oblate)
    neptune_position = np.asarray(state["NEPTUNE"]["position"], dtype=float)
    _, point_observed = lane1.relative_accelerations(
        "NEPTUNE", neptune_position, oblate, metrics, config
    )
    au_km = float(config["constants"]["au_km"])
    delta_km = neptune_position * au_km
    distance_km = float(np.linalg.norm(delta_km))
    expected_point = (
        -float(metrics["A_Q"])
        * _enclosed_fraction_independent(float(np.linalg.norm(neptune_position)) / float(metrics["L_au"]))
        * float(config["gm_km3_s2"]["SUN"])
        * delta_km
        / distance_km**3
        * 1000.0
    )

    _, v1_payloads = _v1_rebuild()
    summary = _csv_rows(v1_payloads["force-summary.csv"])
    _, force_raw = lane1._time_and_force_rows(config)
    d05 = [row for row in force_raw if row["domain_id"] == "D05_SUN_EIGHT_PLANETS"]
    newton_all = np.asarray(
        [[row[f"newton_{axis}_m_s2"] for axis in "xyz"] for row in d05], dtype=float
    )
    dark_all = np.asarray(
        [[row[f"dark_{axis}_m_s2"] for axis in "xyz"] for row in d05], dtype=float
    )
    common_scale = _projection_scale(newton_all, dark_all)
    common_error = _normal_equation_error(
        newton_all, dark_all - common_scale * newton_all
    )
    reported_common = {
        float(row["common_inverse_square_scale"])
        for row in summary
        if row["domain_id"] == "D05_SUN_EIGHT_PLANETS"
    }
    _require(len(reported_common) == 1, "v1 common projection not common")
    scale_errors = [abs(common_scale - reported_common.pop())]
    per_target_errors: dict[str, float] = {}
    per_target_normal_errors: dict[str, float] = {}
    for target in lane1._TARGETS:
        rows = [row for row in d05 if row["target"] == target]
        newton = np.asarray(
            [[row[f"newton_{axis}_m_s2"] for axis in "xyz"] for row in rows], dtype=float
        )
        dark = np.asarray(
            [[row[f"dark_{axis}_m_s2"] for axis in "xyz"] for row in rows], dtype=float
        )
        scale = _projection_scale(newton, dark)
        reported = next(
            float(row["per_target_inverse_square_scale"])
            for row in summary
            if row["domain_id"] == "D05_SUN_EIGHT_PLANETS" and row["target"] == target
        )
        per_target_errors[target] = abs(scale - reported)
        per_target_normal_errors[target] = _normal_equation_error(newton, dark - scale * newton)
        scale_errors.append(abs(scale - reported))

    return {
        "self_force": {
            "cases": self_norms,
            "maximum_norm_m_s2": max(self_norms.values()),
            "earth_EMB_alias_explicitly_checked": True,
        },
        "relative_sun": {
            "independent_manual_cases": cases,
            "maximum_newton_component_error_m_s2": max_newton_error,
            "maximum_dark_component_error_m_s2": max_dark_error,
            "point_sun_closed_form_error_m_s2": float(np.linalg.norm(point_observed - expected_point)),
        },
        "inverse_square_projection": {
            "D05_common_scale": common_scale,
            "common_normal_equation_relative_error": common_error,
            "per_target_normal_equation_relative_errors": per_target_normal_errors,
            "reported_scale_max_absolute_error": max(scale_errors),
            "per_target_reported_scale_absolute_errors": per_target_errors,
            "interpretation": "Unweighted algebraic projections only; no observational fit authority.",
        },
    }


def _report(theorem: Mapping[str, Any], checks: Mapping[str, Any]) -> bytes:
    text = f"""# Corrected Lane 6 GQNS Solar source-domain stress test v2

## Result

**{_STATUS}**. The v1 source-only calculation, its large frozen-trajectory accelerations, nonlinear decomposition, and every source-boundary failure are preserved. The v1 observational exclusion decision and its 21632.16367487 INPOP ratio are withdrawn.

The valid analytic statement is narrower: for fixed positive `A_Q` and finite `L`, the point-source enhancement `A_Q*[1-(1+r/L)exp(-r/L)]` is strictly increasing, so one exact common inverse-square constant cannot match unequal positive radii. The true minimum and maximum pairwise enhancement spreads for the four reported radii are `{theorem['minimum_pairwise_absolute_spread']:.12e}` and `{theorem['maximum_pairwise_absolute_spread']:.12e}`.

## Preserved source-only stress result

For D05 the median frozen moments remain `A_Q={theorem['D05_median_A_Q']:.12g}` and `L={theorem['D05_median_L_au']:.12g} au`. The maximum D05 Neptune common-projection radial stress amplitude remains `{theorem['D05_Neptune_source_only_acceleration']['common_projection_radial_residual_max_m_s2']:.12e} m/s^2`; it is a source-only diagnostic, not an observed or postfit residual and is not divided by the INPOP constant-acceleration threshold.

Only D05-D06-D07 stability under the declared Moon and asteroid refinements is claimed. Host-only, inner-boundary, and remote-source changes remain explicit localization failures.

## Independent controls

- Maximum isolated self-force norm: `{checks['self_force']['maximum_norm_m_s2']:.3e} m/s^2`.
- Maximum independent target-minus-Sun dark-force component disagreement: `{checks['relative_sun']['maximum_dark_component_error_m_s2']:.3e} m/s^2`.
- Maximum reported projection-scale disagreement: `{checks['inverse_square_projection']['reported_scale_max_absolute_error']:.3e}`.
- The projection checks enforce their normal equations but confer no ephemeris-fit authority.

## Matched refit gate

The exact DE440 source identity, integration interval, body inventory, D05-D07 variants, DOP853 solver tolerances, variational fit, common nuisance set, and response gate are frozen in `matched-ephemeris-refit-preflight.json`. Execution remains blocked until the official ephemeris and small-body sources, solver implementation, DE440 replay gates, response manifest, weights, priors, thresholds, and injection fixtures are SHA-256 sealed. No ephemeris binary, observational response, row, or residual value was opened by this builder.

## Claim ceiling

This package is a deterministic analytic and source-only stress test plus an execution-blocked refit preflight. It makes no observational exclusion, preference, DE440 postfit, or INPOP postfit claim.
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
    package_bindings = _validate_package_files()
    theorem = _corrected_theorem(config)
    checks = _independent_checks()
    preflight = config["matched_ephemeris_refit_preflight"]
    payloads = {
        "corrected-source-only-theorem.json": canonical_bytes(theorem),
        "independent-mechanics-projection-checks.json": canonical_bytes(checks),
        "matched-ephemeris-refit-preflight.json": canonical_bytes(preflight),
        "report.md": _report(theorem, checks),
    }
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
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": _PACKAGE_ID,
        "status": _STATUS,
        "decision": _STATUS,
        "observational_decision": theorem["observational_decision"],
        "predecessor_hashes": predecessor,
        "predecessor_receipt_content_sha256": config["supersedes"]["receipt_content_sha256"],
        "package_bindings": package_bindings,
        "summary": {
            "v1_source_only_stress_result_preserved": True,
            "v1_observational_exclusion_withdrawn": True,
            "v1_invalid_ratio_retained_as_failure": theorem["superseded_failure"]["v1_ratio"],
            "true_minimum_pairwise_spread": theorem["minimum_pairwise_absolute_spread"],
            "true_maximum_pairwise_spread": theorem["maximum_pairwise_absolute_spread"],
            "independent_relative_sun_cases": checks["relative_sun"]["independent_manual_cases"],
            "observational_response_rows_opened": 0,
            "parameters_fit_to_responses": 0,
            "refit_execution_authorized": False,
        },
        "retained_failures": {
            "invalid_INPOP_bound_transfer": theorem["superseded_failure"],
            "source_localization_boundaries": theorem["domain_statement"],
            "matched_refit_blockers": preflight["response_gate"]["blockers"],
            "solver_implementation_incomplete": True,
        },
        "access_contract": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
        "artifact_index": artifact_index,
    }
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
        print(json.dumps({"status": receipt["status"], "content_sha256": receipt["content_sha256"]}, sort_keys=True))
    else:
        if _repo_path(OUTPUT_PATH).exists():
            print(_read_json(_repo_path(OUTPUT_PATH), "receipt")["status"])
        else:
            print(load_config()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
