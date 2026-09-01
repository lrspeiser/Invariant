"""Compile fixed source-only matched-acceleration galaxy/cluster predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_gain_persistence_gp01_xcop_source_preflight as xcop

CONFIG_PATH = Path("configs/open_gravity_matched_acceleration_cross_scale_predictions_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_matched_acceleration_cross_scale_predictions_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_matched_acceleration_cross_scale_predictions_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-matched-acceleration-cross-scale-predictions-v1/receipt.json"
)

_CONFIG_RAW_SHA256 = "ee21d151c7f2f970c55ca694e5e5b317e6bdcf6a25c844102c8cdda485dff2ec"
_CONFIG_CONTENT_SHA256 = "bd079fb4ff63ed958df9b5d57fcd9f3970c4d55de47ec4525fdc8506cfede44c"
_MODULE_SEMANTIC_SHA256 = "95df2bf4bbfd66584801ea1233653801956a8935c8c1f6b5b926a0962a9e790a"
_TEST_RAW_SHA256 = "5fc955c7adfbcf0dc3a53126d4a86b50c24dc4641b1481ffe801f785b7de8367"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')
_SCHEMA = "invariant-open-gravity-matched-acceleration-cross-scale-predictions-1.0"
_PREDICTION_SCHEMA = "invariant-open-gravity-matched-acceleration-cross-scale-prediction-ledger-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-matched-acceleration-cross-scale-prediction-receipt-1.0"


class PredictionCompilerError(RuntimeError):
    """Raised when a source or formula contract changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PredictionCompilerError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: Path | str) -> Path:
    root = _root().resolve()
    candidate = (root / relative).resolve()
    _require(candidate == root or root in candidate.parents, "path escaped repository")
    return candidate


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized, count = _MODULE_PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", raw)
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PredictionCompilerError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: dict[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["status"] == "SOURCE_ONLY_FIXED_PREDICTIONS_DEVELOPMENT_ONLY", "status changed")
    program = config["candidate_program"]
    _require(program["compiled_candidate_count"] == 32, "candidate count changed")
    _require(
        program["same_constants_across_every_object"] is True, "object-specific constants enabled"
    )
    _require(program["per_object_parameters"] == 0, "per-object parameters enabled")
    _require(program["response_tuning"] is False, "response tuning enabled")
    _require(program["multiplicity_charge_all_cells"] is True, "multiplicity charge lost")
    _require(len(config["published_formula_anchors"]) == 4, "published anchors changed")
    _require(
        len(config["required_but_not_impersonated_comparators"]) == 7,
        "required comparators changed",
    )
    gate = config["method_admission_gate"]
    _require(
        gate["novel_exact_formula_paper_required"] is False,
        "novel ideas were incorrectly required to have a prior exact paper",
    )
    _require(
        gate["proxy_may_impersonate_published_solver"] is False,
        "proxy impersonation enabled",
    )
    _require(
        gate["response_values_may_repair_or_tune_method"] is False,
        "response tuning enabled through the method gate",
    )
    _require(
        gate["missing_real_source_disposition"] == "SOURCE_BLOCKED"
        and gate["missing_paper_or_analytic_benchmark_disposition"] == "THEORY_ONLY_UNVALIDATED"
        and gate["failed_benchmark_disposition"] == "BENCHMARK_FAILED_RETAINED_NOT_SCORED",
        "method failure dispositions changed",
    )
    _require(
        len(gate["builder_and_operator_requirements"]) == 5
        and len(gate["novel_formula_requirements"]) == 5,
        "method evidence requirements changed",
    )
    evidence = config["current_method_evidence"]
    _require(
        [row["component_id"] for row in evidence]
        == [
            "MODEL_LIFTED_GALAXY_2P5D_SOURCE_BUILDER",
            "XCOP_SPHERICAL_SOURCE_RECONSTRUCTION",
            "PUBLISHED_GRAVITY_CONTROLS",
            "NEW_DRIVER_AND_INTERACTION_HYPOTHESES",
        ],
        "method evidence inventory changed",
    )
    _require(
        all(
            row["real_data"]
            and row["primary_or_analytic_anchors"]
            and row["required_passed_checks"]
            and row["disposition"]
            in {
                "BENCHMARKED_SOURCE_READY",
                "BENCHMARKED_CONTROL_READY",
                "FALSIFIABLE_NOVEL_HYPOTHESES_READY_FOR_FIXED_PREDICTION",
            }
            for row in evidence
        ),
        "method evidence became incomplete",
    )
    boundary = config["scientific_boundary"]
    _require(boundary["galaxy_source_profiles_opened"] == 225, "galaxy source count changed")
    _require(boundary["cluster_source_files_opened"] == 13, "cluster source count changed")
    _require(boundary["cluster_response_files_opened"] == 0, "cluster response opened")
    _require(boundary["galaxy_response_rows_opened"] == 0, "galaxy response opened")
    _require(boundary["scores_computed"] == 0 and boundary["models_fit"] == 0, "science executed")
    _require(boundary["network_calls"] == 0, "network enabled")
    claims = config["claims"]
    _require(claims["fixed_source_only_predictions_compiled"] is True, "prediction claim lost")
    _require(claims["published_controls_exactly_labeled"] is True, "control label claim lost")
    _require(
        claims["new_power_laws_are_phenomenological_not_novelty_claims"] is True,
        "novelty caveat lost",
    )
    _require(
        not any(
            value
            for key, value in claims.items()
            if key
            not in {
                "fixed_source_only_predictions_compiled",
                "published_controls_exactly_labeled",
                "new_power_laws_are_phenomenological_not_novelty_claims",
            }
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


def _verify_file_binding(binding: dict[str, Any], role: str) -> None:
    path = _repo_path(binding[f"{role}_path"])
    _require(file_sha256(path) == binding[f"{role}_raw_sha256"], f"{role} binding changed")


def _verify_sources(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    galaxy = config["source_bindings"]["galaxy_builder"]
    for role in ("config", "module", "test", "receipt"):
        _verify_file_binding(galaxy, role)
    profiles_path = _repo_path(galaxy["profiles_path"])
    _require(file_sha256(profiles_path) == galaxy["profiles_raw_sha256"], "galaxy profiles changed")
    profiles = _read_json(profiles_path, "galaxy profiles")
    _require(
        profiles["content_sha256"] == galaxy["profiles_content_sha256"],
        "galaxy profile content changed",
    )
    galaxy_receipt = _read_json(_repo_path(galaxy["receipt_path"]), "galaxy receipt")
    _require(
        galaxy_receipt["content_sha256"] == galaxy["receipt_content_sha256"],
        "galaxy receipt content changed",
    )

    cluster = config["source_bindings"]["xcop_source_preflight"]
    for role in ("config", "module", "test", "receipt"):
        _verify_file_binding(cluster, role)
    cluster_receipt = _read_json(_repo_path(cluster["receipt_path"]), "X-COP source receipt")
    _require(
        cluster_receipt["content_sha256"] == cluster["receipt_content_sha256"],
        "X-COP receipt content changed",
    )

    prior = config["source_bindings"]["prior_art"]
    for role in ("config", "module", "test", "receipt"):
        _verify_file_binding(prior, role)
    prior_receipt = _read_json(_repo_path(prior["receipt_path"]), "prior-art receipt")
    _require(
        prior_receipt["content_sha256"] == prior["receipt_content_sha256"],
        "prior-art receipt content changed",
    )
    for role in ("config", "module", "test", "receipt"):
        relative = prior[f"{role}_path"]
        result = subprocess.run(
            ["git", "show", f"{prior['commit']}:{relative}"],
            cwd=_root(),
            check=True,
            capture_output=True,
        )
        _require(
            hashlib.sha256(result.stdout).hexdigest() == prior[f"{role}_raw_sha256"],
            "prior-art commit binding changed",
        )
    return profiles, cluster_receipt


def mond_standard(g_newton: float, a0: float) -> float:
    _require(
        math.isfinite(g_newton) and g_newton >= 0.0 and math.isfinite(a0) and a0 > 0.0,
        "invalid MOND inputs",
    )
    if g_newton == 0.0:
        return 0.0
    g2 = 0.5 * (g_newton * g_newton + math.sqrt(g_newton**4 + 4.0 * g_newton * g_newton * a0 * a0))
    return math.sqrt(g2)


def rar_2016(g_bar: float, a0: float) -> float:
    _require(g_bar >= 0.0 and a0 > 0.0, "invalid RAR inputs")
    if g_bar == 0.0:
        return 0.0
    denominator = -math.expm1(-math.sqrt(g_bar / a0))
    return g_bar / denominator


def emond_a0(config: dict[str, Any], potential_depth_c2: float) -> float:
    constants = config["constants"]
    total_depth = float(constants["emond_external_potential_floor_c2"]) + potential_depth_c2
    p = total_depth / float(constants["emond_beta"]) ** 2
    mu_p = p / math.sqrt(1.0 + p * p)
    return float(constants["emond_cH0_m_s2"]) * math.sqrt(mu_p)


def _number_id(value: float) -> str:
    sign = "P" if value >= 0.0 else "M"
    return f"{sign}{abs(value):.6g}".replace(".", "P").replace("-", "M")


def candidate_registry(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": "GR_BARYONS_WEAK_FIELD",
            "kind": "PUBLISHED_CONTROL",
            "program": "g=g_b",
            "paper": "weak-field GR/Newton",
        },
        {
            "candidate_id": "RAR_2016_EMPIRICAL",
            "kind": "PUBLISHED_CONTROL",
            "program": "g=g_b/[1-exp(-sqrt(g_b/a0))]",
            "paper": "RAR_2016",
        },
        {
            "candidate_id": "MOND_STANDARD_MU",
            "kind": "PUBLISHED_CONTROL",
            "program": "mu(g/a0)g=g_b; mu(x)=x/sqrt(1+x^2)",
            "paper": "MILGROM_1983",
        },
        {
            "candidate_id": "EMOND_2012_APPROX_T2_OMITTED",
            "kind": "PUBLISHED_APPROXIMATE_CONTROL",
            "program": "EMOND Eq.6+Eq.7 algebraic approximation; T2 omitted",
            "paper": "EMOND_2012",
        },
    ]
    for driver, key in (
        ("POTENTIAL", "potential_power_exponents"),
        ("DENSITY", "density_power_exponents"),
        ("TIDAL", "tidal_power_exponents"),
    ):
        for exponent in config["candidate_program"][key]:
            rows.append(
                {
                    "candidate_id": f"{driver}_POWER_{_number_id(float(exponent))}",
                    "kind": "NEW_PHENOMENOLOGICAL_ABLATION",
                    "driver": driver,
                    "exponent": float(exponent),
                    "program": "a_eff=a0*clip(driver/reference,1/16,16)^exponent; standard-mu solve",
                }
            )
    for coefficient in config["candidate_program"]["geometry_exponential_coefficients"]:
        rows.append(
            {
                "candidate_id": f"GEOMETRY_EXP_{_number_id(float(coefficient))}",
                "kind": "NEW_PHENOMENOLOGICAL_ABLATION",
                "driver": "GEOMETRY",
                "exponent": float(coefficient),
                "program": "a_eff=a0*clip(exp(k*(A-Aref)),1/16,16); standard-mu solve",
            }
        )
    for index, cell in enumerate(config["candidate_program"]["interaction_cells"], start=1):
        rows.append(
            {
                "candidate_id": f"INTERACTION_{index:02d}_{cell['driver_a']}_{cell['driver_b']}",
                "kind": "NEW_PHENOMENOLOGICAL_INTERACTION",
                "drivers": [cell["driver_a"], cell["driver_b"]],
                "exponents": [float(cell["exponent_a"]), float(cell["exponent_b"])],
                "program": "multiply the two frozen driver factors; standard-mu solve",
            }
        )
    for row in rows:
        if row["kind"].startswith("PUBLISHED"):
            row["admission_evidence"] = {
                "class": "PRIMARY_PUBLISHED_FORMULA",
                "real_source_domains": ["GALAXY_2P5D", "XCOP_SPHERICAL"],
                "required_checks": [
                    "equation_residual",
                    "deep_or_weak_field_limit",
                    "high_acceleration_limit",
                ],
            }
        else:
            drivers = [row["driver"]] if "driver" in row else row["drivers"]
            row["admission_evidence"] = {
                "class": "NOVEL_FALSIFIABLE_HYPOTHESIS",
                "real_source_domains": (
                    ["GALAXY_2P5D"] if "GEOMETRY" in drivers else ["GALAXY_2P5D", "XCOP_SPHERICAL"]
                ),
                "required_checks": [
                    "driver_present_in_frozen_source",
                    "zero_coupling_recovers_MOND_standard_mu",
                    "finite_positive_prediction",
                    "high_acceleration_solar_recovery",
                ],
            }
    _require(
        len(rows) == int(config["candidate_program"]["compiled_candidate_count"]),
        "candidate registry count changed",
    )
    _require(len({row["candidate_id"] for row in rows}) == len(rows), "candidate IDs collided")
    return rows


def _galaxy_rows(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    conversion = 1.98847e30 / 3.085677581491367e16**3
    rows: list[dict[str, Any]] = []
    for object_row in profiles["objects"]:
        for cell in object_row["cell_profiles"]:
            for radial in cell["radial_profile"]:
                rows.append(
                    {
                        "domain": "GALAXY",
                        "object_id": object_row["object_id"],
                        "source_cell_id": cell["cell_id"],
                        "radius_kpc": radial["radius_kpc"],
                        "g_b_m_s2": radial["g_b_m_s2"],
                        "potential_depth_c2": radial["potential_depth_c2"],
                        "density_kg_m3": radial["rho_midplane_msun_pc3"] * conversion,
                        "tidal_s2": radial["tidal_frobenius_s2"],
                        "geometry_asymmetry": radial["radial_force_rms_asymmetry"],
                    }
                )
    _require(len(rows) == 225 * 60, "galaxy source-row count changed")
    return rows


def _spherical_potential(
    gravity: float, radius_m: np.ndarray, mass_kg: np.ndarray, c_m_s: float
) -> np.ndarray:
    shell = 0.0
    potential = np.empty_like(radius_m)
    potential[-1] = -gravity * mass_kg[-1] / radius_m[-1]
    for index in range(radius_m.size - 2, -1, -1):
        delta_mass = max(float(mass_kg[index + 1] - mass_kg[index]), 0.0)
        shell += delta_mass * 0.5 * (1.0 / radius_m[index] + 1.0 / radius_m[index + 1])
        potential[index] = -gravity * (mass_kg[index] / radius_m[index] + shell)
    return np.abs(potential) / (c_m_s * c_m_s)


def _cluster_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = _root()
    source_config = xcop.load_config(root)
    raw_root, records, item59 = xcop._source_records(root, source_config)
    by_key = {(str(row["cluster"]), str(row["role"])): row for row in records}
    rows: list[dict[str, Any]] = []
    for cluster in source_config["population"]["development_clusters"]:

        def payload(role: str, cluster_id: str = cluster) -> bytes:
            record = by_key[(cluster_id, role)]
            value = (raw_root / str(record["member"])).read_bytes()
            _require(len(value) == int(record["bytes"]), "cluster source bytes changed")
            _require(
                hashlib.sha256(value).hexdigest() == record["sha256"], "cluster source hash changed"
            )
            return value

        density, header = xcop._fits_table(
            payload("density"), int(source_config["source_contract"]["density_hdu"])
        )
        r500 = float(header["R500"])
        radius_kpc = np.asarray(density["RW_X"], dtype=float) * r500
        constants = item59["constants"]
        radius_m = radius_kpc * float(constants["kiloparsec_m"])
        gas_density = (
            np.maximum(np.asarray(density["NE"], dtype=float), np.finfo(float).tiny)
            * 1.0e6
            * float(constants["mean_molecular_weight_per_electron"])
            * float(constants["proton_mass_kg"])
        )
        gas_mass = xcop._cumulative_mass(radius_m, gas_density)
        stellar = None
        if (cluster, "stellar_mass") in by_key:
            stellar_table, _ = xcop._fits_table(
                payload("stellar_mass"), int(source_config["source_contract"]["stellar_hdu"])
            )
            stellar = {
                "radius_kpc": np.asarray(stellar_table["RADIUS"], dtype=float),
                "mass_msun": np.asarray(stellar_table["MSTAR"], dtype=float),
                "mass_low_msun": np.asarray(stellar_table["MSTAR_LO"], dtype=float),
                "mass_high_msun": np.asarray(stellar_table["MSTAR_HI"], dtype=float),
            }
        variant = {
            "nuisances": {
                "published_stellar_mass_scale": 1.0,
                "missing_stellar_to_gas_mass_ratio": 0.1,
            }
        }
        stellar_mass = xcop._member_mass(
            {"stellar": stellar}, radius_kpc, gas_mass, variant, "nominal", item59
        )
        baryonic_mass = gas_mass + stellar_mass
        gravity = float(constants["gravity_si"])
        g_b = gravity * baryonic_mass / radius_m**2
        effective_density = xcop._effective_density(radius_m, baryonic_mass)
        tidal = math.sqrt(2.0 / 3.0) * np.abs(
            4.0 * math.pi * gravity * effective_density - 3.0 * g_b / radius_m
        )
        potential = _spherical_potential(
            gravity, radius_m, baryonic_mass, float(config["constants"]["c_m_s"])
        )
        for index in range(radius_kpc.size):
            rows.append(
                {
                    "domain": "CLUSTER",
                    "object_id": cluster,
                    "source_cell_id": "XCOP_NOMINAL_SPHERICAL",
                    "radius_kpc": float(radius_kpc[index]),
                    "g_b_m_s2": float(g_b[index]),
                    "potential_depth_c2": float(potential[index]),
                    "density_kg_m3": float(effective_density[index]),
                    "tidal_s2": float(tidal[index]),
                    "geometry_asymmetry": None,
                }
            )
    _require(
        {row["object_id"] for row in rows}
        == set(source_config["population"]["development_clusters"]),
        "cluster population changed",
    )
    return rows


def _references(config: dict[str, Any], galaxy_rows: list[dict[str, Any]]) -> dict[str, float]:
    selected: list[dict[str, Any]] = []
    a0 = float(config["constants"]["a0_m_s2"])
    for object_id in ("NGC2903", "NGC3351", "NGC3627"):
        candidates = [
            row
            for row in galaxy_rows
            if row["object_id"] == object_id
            and row["source_cell_id"]
            == "ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.136986301369863:HG200"
        ]
        selected.append(
            min(candidates, key=lambda row: abs(math.log(max(row["g_b_m_s2"], 1.0e-30) / a0)))
        )
    return {
        "potential_depth_c2": float(np.median([row["potential_depth_c2"] for row in selected])),
        "density_kg_m3": float(np.median([row["density_kg_m3"] for row in selected])),
        "tidal_s2": float(np.median([row["tidal_s2"] for row in selected])),
        "geometry_asymmetry": float(np.median([row["geometry_asymmetry"] for row in selected])),
    }


def _driver_factor(
    config: dict[str, Any],
    row: dict[str, Any],
    references: dict[str, float],
    driver: str,
    exponent: float,
) -> float | None:
    floor = float(config["constants"]["driver_ratio_floor"])
    ceiling = float(config["constants"]["driver_ratio_ceiling"])
    if driver == "GEOMETRY":
        if row["geometry_asymmetry"] is None:
            return None
        factor = math.exp(
            exponent * (float(row["geometry_asymmetry"]) - references["geometry_asymmetry"])
        )
    else:
        field = {
            "POTENTIAL": "potential_depth_c2",
            "DENSITY": "density_kg_m3",
            "TIDAL": "tidal_s2",
        }[driver]
        ratio = min(max(float(row[field]) / references[field], floor), ceiling)
        factor = ratio**exponent
    return min(max(factor, floor), ceiling)


def predict(
    config: dict[str, Any],
    candidate: dict[str, Any],
    row: dict[str, Any],
    references: dict[str, float],
) -> tuple[str, float | None, float | None]:
    g_b = float(row["g_b_m_s2"])
    a0 = float(config["constants"]["a0_m_s2"])
    candidate_id = candidate["candidate_id"]
    if candidate_id == "GR_BARYONS_WEAK_FIELD":
        return "COMPILED", g_b, 0.0
    if candidate_id == "RAR_2016_EMPIRICAL":
        return "COMPILED", rar_2016(g_b, a0), a0
    if candidate_id == "MOND_STANDARD_MU":
        return "COMPILED", mond_standard(g_b, a0), a0
    if candidate_id == "EMOND_2012_APPROX_T2_OMITTED":
        effective = emond_a0(config, float(row["potential_depth_c2"]))
        return "COMPILED", mond_standard(g_b, effective), effective
    if "driver" in candidate:
        factor = _driver_factor(
            config, row, references, candidate["driver"], float(candidate["exponent"])
        )
        if factor is None:
            return "SOURCE_BLOCKED_DRIVER_UNAVAILABLE", None, None
    else:
        factors = [
            _driver_factor(config, row, references, driver, exponent)
            for driver, exponent in zip(candidate["drivers"], candidate["exponents"], strict=True)
        ]
        if any(factor is None for factor in factors):
            return "SOURCE_BLOCKED_DRIVER_UNAVAILABLE", None, None
        factor = math.prod(float(value) for value in factors)
        factor = min(max(factor, 1.0 / 256.0), 256.0)
    effective = a0 * factor
    return "COMPILED", mond_standard(g_b, effective), effective


def build_predictions(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    galaxy_profiles, _cluster_receipt = _verify_sources(config)
    candidates = candidate_registry(config)
    source_rows = _galaxy_rows(galaxy_profiles) + _cluster_rows(config)
    references = _references(config, source_rows)
    prediction_rows: list[dict[str, Any]] = []
    compiled = 0
    blocked = 0
    for source in source_rows:
        predictions: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            disposition, g_prediction, effective_a0 = predict(config, candidate, source, references)
            predictions[candidate["candidate_id"]] = {
                "disposition": disposition,
                "g_prediction_m_s2": g_prediction,
                "effective_a0_m_s2": effective_a0,
            }
            compiled += disposition == "COMPILED"
            blocked += disposition != "COMPILED"
        prediction_rows.append({"source": source, "predictions": predictions})
    worst_a0 = float(config["constants"]["a0_m_s2"]) * 256.0
    solar_g = float(config["constants"]["solar_benchmark_g_m_s2"])
    solar_fraction = abs(mond_standard(solar_g, worst_a0) / solar_g - 1.0)
    _require(
        solar_fraction < float(config["constants"]["solar_max_fractional_deviation"]),
        "high-acceleration recovery failed",
    )
    ledger: dict[str, Any] = {
        "schema": _PREDICTION_SCHEMA,
        "package_id": config["package_id"],
        "candidate_registry": candidates,
        "candidate_registry_sha256": content_sha256(candidates),
        "method_admission_gate": config["method_admission_gate"],
        "current_method_evidence": config["current_method_evidence"],
        "method_evidence_sha256": content_sha256(
            {
                "gate": config["method_admission_gate"],
                "evidence": config["current_method_evidence"],
            }
        ),
        "reference_values": references,
        "reference_rule": config["reference_rule"],
        "source_row_count": len(source_rows),
        "galaxy_source_row_count": sum(
            row["source"]["domain"] == "GALAXY" for row in prediction_rows
        ),
        "cluster_source_row_count": sum(
            row["source"]["domain"] == "CLUSTER" for row in prediction_rows
        ),
        "prediction_rows": prediction_rows,
        "compiled_prediction_count": compiled,
        "blocked_prediction_count": blocked,
        "solar_worst_case_fractional_deviation": solar_fraction,
        "required_but_not_impersonated_comparators": config[
            "required_but_not_impersonated_comparators"
        ],
        "scientific_boundary": config["scientific_boundary"],
    }
    ledger["prediction_row_root_sha256"] = content_sha256(prediction_rows)
    ledger["content_sha256"] = content_sha256(ledger)
    return ledger


def build_packet(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = build_predictions(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "FIXED_SOURCE_ONLY_PREDICTIONS_READY_FOR_DEVELOPMENT_RESPONSE_SCORING",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "source_bindings": config["source_bindings"],
        "published_formula_anchors": config["published_formula_anchors"],
        "candidate_count": len(ledger["candidate_registry"]),
        "candidate_registry_sha256": ledger["candidate_registry_sha256"],
        "method_admission_gate": ledger["method_admission_gate"],
        "current_method_evidence": ledger["current_method_evidence"],
        "method_evidence_sha256": ledger["method_evidence_sha256"],
        "reference_values": ledger["reference_values"],
        "source_row_count": ledger["source_row_count"],
        "galaxy_source_row_count": ledger["galaxy_source_row_count"],
        "cluster_source_row_count": ledger["cluster_source_row_count"],
        "compiled_prediction_count": ledger["compiled_prediction_count"],
        "blocked_prediction_count": ledger["blocked_prediction_count"],
        "prediction_row_root_sha256": ledger["prediction_row_root_sha256"],
        "private_prediction_path": config["private_prediction_output_path"],
        "private_prediction_raw_sha256": hashlib.sha256(canonical_bytes(ledger)).hexdigest(),
        "private_prediction_content_sha256": ledger["content_sha256"],
        "solar_worst_case_fractional_deviation": ledger["solar_worst_case_fractional_deviation"],
        "required_but_not_impersonated_comparators": config[
            "required_but_not_impersonated_comparators"
        ],
        "scientific_boundary": config["scientific_boundary"],
        "claims": config["claims"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return ledger, receipt


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
    ledger, receipt = build_packet(config)
    ledger_status = _atomic_no_clobber(
        _repo_path(config["private_prediction_output_path"]), canonical_bytes(ledger)
    )
    receipt_status = _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))
    return "CREATED" if "CREATED" in {ledger_status, receipt_status} else "EXISTING_IDENTICAL"


def check_packet() -> str:
    config = load_config()
    ledger, receipt = build_packet(config)
    _require(
        _repo_path(config["private_prediction_output_path"]).read_bytes()
        == canonical_bytes(ledger),
        "prediction ledger does not rebuild",
    )
    _require(
        _repo_path(OUTPUT_PATH).read_bytes() == canonical_bytes(receipt), "receipt does not rebuild"
    )
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
        if _repo_path(OUTPUT_PATH).exists():
            receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
            print(
                json.dumps(
                    {
                        "status": receipt["status"],
                        "decision": receipt["decision"],
                        "candidates": receipt["candidate_count"],
                        "responses": receipt["scientific_boundary"]["galaxy_response_rows_opened"],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    {"status": load_config()["status"], "decision": "NOT_WRITTEN"}, sort_keys=True
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
