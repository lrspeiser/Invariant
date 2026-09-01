"""Causal differential-propagation kernel with response-free falsifiers."""

from __future__ import annotations

import argparse
import cmath
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/open_gravity_differential_propagation_kernel_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_differential_propagation_kernel_v1.py")
TEST_PATH = Path("tests/test_open_gravity_differential_propagation_kernel_v1.py")
OUTPUT_PATH = Path("runs/gravity/theory/open-gravity-differential-propagation-kernel-v1.json")
CONFIG_SCHEMA = "invariant-open-gravity-differential-propagation-kernel-config-1.0"
RECEIPT_SCHEMA = "invariant-open-gravity-differential-propagation-kernel-receipt-1.0"
DECISION = (
    "PASS_CAUSAL_RETARDED_KERNEL_AND_TARGET_FREE_RECOVERY_"
    "STATIC_SPEED_ONLY_ENHANCEMENT_EXACTLY_FALSIFIED_REAL_DATA_PREFLIGHT_FROZEN"
)

EXPECTED_CONFIG_RAW_SHA256 = "9a23fc7a011086334a20e81b2f1cbc25097847bced8105ca3ee40fb5af54e0db"
EXPECTED_MODULE_SEMANTIC_SHA256 = "c7986af32ab5c8b8c9e0c82d9e8486efd9f423f9e64f98921ae9e9684bd91ade"
EXPECTED_TEST_RAW_SHA256 = "c391810c7af86bce496926bbcc672a07ff86c1b770f397835020883168bf2696"


class DifferentialPropagationError(RuntimeError):
    """Raised when a propagation, scope, or integrity gate fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _sha256_bytes(_canonical_bytes(body))


def _module_semantic_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'EXPECTED_MODULE_SEMANTIC_SHA256 = (?:"[0-9a-f]{64}"|"__MODULE_SEMANTIC_SHA256__")',
        'EXPECTED_MODULE_SEMANTIC_SHA256 = "<SELF>"',
        text,
        count=1,
    )
    return _sha256_bytes(normalized.encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DifferentialPropagationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DifferentialPropagationError("JSON root must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DifferentialPropagationError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "open-gravity-differential-propagation-kernel-v1",
        "analysis identity changed",
    )
    _require(
        config.get("package")
        == {
            "module_path": MODULE_PATH.as_posix(),
            "test_path": TEST_PATH.as_posix(),
            "output_path": OUTPUT_PATH.as_posix(),
        },
        "package paths changed",
    )
    law = config.get("law")
    _require(isinstance(law, dict), "law missing")
    _require(
        law.get("field_equation", "").startswith("d_t^2 Phi + 2 Gamma"),
        "field law changed",
    )
    _require(
        law.get("photon_control") == "k_gamma=omega/c and t_gamma=D/c",
        "photon control changed",
    )
    assumptions = config.get("assumptions")
    _require(
        isinstance(assumptions, dict)
        and assumptions.get("clock", "").startswith("A single asymptotic observer clock")
        and assumptions.get("conservation", "").startswith("For Gamma=0"),
        "clock or conservation assumptions changed",
    )
    _require(
        [item.get("id") for item in config.get("derived_observables", [])]
        == [
            "O1_COMMON_SOURCE_ARRIVAL_DELAY",
            "O2_PROPAGATION_PHASE_CURVATURE",
            "O3_ATTENUATION_SLOPE",
            "O4_SOURCE_FORCE_CROSS_PHASE",
        ],
        "observable inventory changed",
    )
    theorems = config.get("theorems_and_falsifiers")
    _require(
        isinstance(theorems, dict)
        and "independent of c_g" in theorems.get("static_invariance", "")
        and "fails the closed-system conservation gate"
        in theorems.get("attenuation_conservation", "")
        and "not available without a covariant" in theorems.get("solar", "")
        and "cannot be claimed" in theorems.get("binary", ""),
        "falsifier scope changed",
    )
    injections = config.get("target_free_injections")
    _require(
        isinstance(injections, list)
        and len(injections) == 2
        and [item.get("id") for item in injections]
        == ["INJ_CONSERVATIVE_DISPERSIVE", "INJ_ATTENUATING_SUPERLUMINAL"],
        "injection inventory changed",
    )
    benchmarks = config.get("frozen_published_benchmarks")
    _require(
        isinstance(benchmarks, list)
        and [item.get("id") for item in benchmarks]
        == [
            "GW170817_GRB170817A_SPEED",
            "GWTC3_IMPROVED_DISPERSION_2026",
            "B1913_ORBITAL_DECAY",
            "DOUBLE_PULSAR_QUADRUPOLE_DAMPING",
        ],
        "benchmark inventory changed",
    )
    benchmark_by_id = {item["id"]: item for item in benchmarks}
    _require(
        benchmark_by_id["GW170817_GRB170817A_SPEED"]["value"]
        == {
            "delay_seconds": 1.74,
            "delay_sigma_seconds": 0.05,
            "delta_c_over_c_min": -3e-15,
            "delta_c_over_c_max": 7e-16,
        },
        "GW170817 benchmark changed",
    )
    _require(
        benchmark_by_id["GWTC3_IMPROVED_DISPERSION_2026"]["value"]
        == {"graviton_mass_90pct_upper_eV_c2": 2.21e-23},
        "mass benchmark changed",
    )
    preflight = config.get("public_data_preflight")
    _require(
        isinstance(preflight, dict)
        and preflight.get("state") == "SOURCE_READY_NOT_OPENED_BY_BUILDER"
        and preflight.get("dataset_doi") == "10.7935/K5B8566F"
        and preflight.get("exact_product", "").startswith("4096-second 4096-Hz HDF5"),
        "public data preflight changed",
    )
    _require(len(config.get("novelty_neighbors", [])) == 4, "novelty inventory changed")
    _require(len(config.get("branch_catalog", [])) == 8, "branch inventory changed")
    _require(
        config.get("claim_boundary")
        == {
            "causal_retarded_kernel_derived": True,
            "target_free_injection_recovery_passed": True,
            "static_speed_only_enhancement_falsified": True,
            "published_benchmarks_frozen_not_fit": True,
            "real_observational_rows_scored": False,
            "covariant_tensor_completion": False,
            "solar_moving_body_prediction": False,
            "binary_radiation_reaction_prediction": False,
            "galaxy_or_cluster_anomaly_explained": False,
            "historical_novelty_established": False,
            "publication_ready": False,
        },
        "claim boundary changed",
    )
    _require(
        config.get("access_ledger")
        == {
            "observational_files_opened": 0,
            "observational_rows_read": 0,
            "real_scores_computed": 0,
            "network_calls_by_builder": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "access ledger changed",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / CONFIG_PATH).resolve()
    _require(_sha256_file(path) == EXPECTED_CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _validate_local_integrity(base: Path) -> dict[str, str]:
    module = (base / MODULE_PATH).resolve()
    test = (base / TEST_PATH).resolve()
    _require(module == Path(__file__).resolve(), "module path changed")
    semantic = _module_semantic_sha256(module)
    _require(semantic == EXPECTED_MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(_sha256_file(test) == EXPECTED_TEST_RAW_SHA256, "test bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(module),
        "module_semantic_sha256": semantic,
        "test_raw_sha256": _sha256_file(test),
    }


def analytic_controls() -> dict[str, dict[str, Any]]:
    omega, gamma, c_g, k, mu, zeta, k_star, grav = sp.symbols(
        "omega Gamma c_g k mu zeta k_star G", positive=True
    )
    phi, velocity = sp.symbols("Phi v", real=True)
    spatial = k**2 + mu**2 + zeta * k**4 / k_star**2
    denominator = -(omega**2) - 2 * sp.I * gamma * omega + c_g**2 * spatial
    transfer = -4 * sp.pi * grav * c_g**2 / denominator
    static_massless = sp.simplify(transfer.subs({omega: 0, mu: 0, zeta: 0}))
    gr_control = sp.simplify(
        denominator.subs({gamma: 0, mu: 0, zeta: 0, c_g: 1}) - (k**2 - omega**2)
    )
    energy = (velocity**2 / c_g**2 + spatial * phi**2) / (8 * sp.pi * grav)
    acceleration = -2 * gamma * velocity - c_g**2 * spatial * phi
    energy_rate = sp.simplify(
        sp.diff(energy, phi) * velocity + sp.diff(energy, velocity) * acceleration
    )
    expected_rate = -gamma * velocity**2 / (2 * sp.pi * grav * c_g**2)
    laplace_s = sp.symbols("s")
    pole_product = sp.expand(
        (laplace_s + gamma - sp.sqrt(gamma**2 - c_g**2 * spatial))
        * (laplace_s + gamma + sp.sqrt(gamma**2 - c_g**2 * spatial))
    )
    checks = {
        "A01_STATIC_POISSON_INDEPENDENT_OF_CG": {
            "passed": sp.simplify(static_massless + 4 * sp.pi * grav / k**2) == 0,
            "residual": str(sp.simplify(static_massless + 4 * sp.pi * grav / k**2)),
        },
        "A02_STATIC_DAMPING_DROPS_OUT": {
            "passed": sp.diff(transfer.subs(omega, 0), gamma) == 0,
            "residual": str(sp.diff(transfer.subs(omega, 0), gamma)),
        },
        "A03_LUMINAL_MASSLESS_CONTROL": {
            "passed": gr_control == 0,
            "residual": str(gr_control),
        },
        "A04_CONSERVATIVE_ENERGY_IDENTITY": {
            "passed": sp.simplify(energy_rate - expected_rate) == 0,
            "residual": str(sp.simplify(energy_rate - expected_rate)),
        },
        "A05_ZERO_DAMPING_CONSERVES_FREE_ENERGY": {
            "passed": sp.simplify(energy_rate.subs(gamma, 0)) == 0,
            "residual": str(sp.simplify(energy_rate.subs(gamma, 0))),
        },
        "A06_DAMPING_REQUIRES_ENERGY_SINK": {
            "passed": sp.simplify(energy_rate / expected_rate) == 1,
            "residual": str(sp.simplify(energy_rate / expected_rate - 1)),
        },
        "A07_RETARDED_POLE_FACTORIZATION": {
            "passed": sp.simplify(
                pole_product - (laplace_s**2 + 2 * gamma * laplace_s + c_g**2 * spatial)
            )
            == 0,
            "residual": str(
                sp.simplify(
                    pole_product - (laplace_s**2 + 2 * gamma * laplace_s + c_g**2 * spatial)
                )
            ),
        },
        "A08_QUARTIC_PHASE_SPEED_UNBOUNDED": {
            "passed": sp.limit(c_g * sp.sqrt(spatial) / k, k, sp.oo) == sp.oo,
            "residual": str(sp.limit(c_g * sp.sqrt(spatial) / k, k, sp.oo)),
        },
    }
    _require(all(item["passed"] for item in checks.values()), "analytic control failed")
    return checks


def radiative_wavenumber(
    omega: float,
    *,
    c_g: float,
    gamma: float,
    mu: float,
    zeta: float,
    k_star: float,
) -> complex:
    _require(omega > 0.0, "omega must be positive")
    _require(c_g > 0.0, "c_g must be positive")
    _require(gamma >= 0.0 and mu >= 0.0 and zeta >= 0.0, "passive parameters required")
    _require(k_star > 0.0, "k_star must be positive")
    driving = (omega**2 + 2j * gamma * omega) / c_g**2 - mu**2
    if zeta == 0.0:
        q_value = driving
    else:
        q_value = (k_star**2 / (2.0 * zeta)) * (
            -1.0 + cmath.sqrt(1.0 + 4.0 * zeta * driving / k_star**2)
        )
    wave_number = cmath.sqrt(q_value)
    if wave_number.imag < 0.0 or (
        math.isclose(wave_number.imag, 0.0, abs_tol=1.0e-15) and wave_number.real < 0.0
    ):
        wave_number = -wave_number
    return wave_number


def propagation_transfer(
    omega: float,
    distance: float,
    *,
    c_g: float,
    gamma: float,
    mu: float,
    zeta: float,
    k_star: float,
) -> complex:
    _require(distance > 0.0, "distance must be positive")
    wave_number = radiative_wavenumber(
        omega,
        c_g=c_g,
        gamma=gamma,
        mu=mu,
        zeta=zeta,
        k_star=k_star,
    )
    return cmath.exp(1j * wave_number * distance) / distance


def _candidate_value(true_value: float, offset: float, zero_scale: float | None) -> float:
    if true_value == 0.0:
        _require(zero_scale is not None, "zero parameter scale missing")
        return offset * zero_scale
    return true_value * (1.0 + offset)


def injection_recovery(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    offsets = config["recovery_grid_fractional_offsets"]
    zero_scales = config["recovery_zero_scales"]
    records: list[dict[str, Any]] = []
    for injection in config["target_free_injections"]:
        observed = [
            propagation_transfer(
                frequency,
                injection["distance"],
                c_g=injection["c_g"],
                gamma=injection["Gamma"],
                mu=injection["mu"],
                zeta=injection["zeta"],
                k_star=injection["k_star"],
            )
            for frequency in injection["frequencies"]
        ]
        candidates: list[tuple[float, dict[str, float]]] = []
        for dc, dg, dm, dz in itertools.product(
            offsets["c_g"], offsets["Gamma"], offsets["mu"], offsets["zeta"]
        ):
            parameters = {
                "c_g": _candidate_value(injection["c_g"], dc, None),
                "Gamma": _candidate_value(injection["Gamma"], dg, zero_scales["Gamma"]),
                "mu": _candidate_value(injection["mu"], dm, zero_scales["mu"]),
                "zeta": _candidate_value(injection["zeta"], dz, zero_scales["zeta"]),
            }
            if (
                parameters["c_g"] <= 0.0
                or parameters["Gamma"] < 0.0
                or parameters["mu"] < 0.0
                or parameters["zeta"] < 0.0
            ):
                continue
            predicted = [
                propagation_transfer(
                    frequency,
                    injection["distance"],
                    c_g=parameters["c_g"],
                    gamma=parameters["Gamma"],
                    mu=parameters["mu"],
                    zeta=parameters["zeta"],
                    k_star=injection["k_star"],
                )
                for frequency in injection["frequencies"]
            ]
            residual = math.fsum(abs(left - right) ** 2 for left, right in zip(predicted, observed))
            candidates.append((residual, parameters))
        candidates.sort(key=lambda item: (item[0], tuple(item[1].values())))
        best_residual, best = candidates[0]
        truth = {key: injection[key] for key in ("c_g", "Gamma", "mu", "zeta")}
        exact = all(math.isclose(best[key], truth[key], abs_tol=1.0e-15) for key in truth)
        _require(exact and best_residual < 1.0e-25, "target-free recovery failed")
        records.append(
            {
                "injection_id": injection["id"],
                "truth": truth,
                "recovered": best,
                "candidate_count": len(candidates),
                "complex_residual": best_residual,
                "exact_recovery": exact,
                "observational_targets_used": 0,
            }
        )
    return records


def benchmark_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["id"]: item for item in config["frozen_published_benchmarks"]}


def classify_branches(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    benchmarks = benchmark_map(config)
    speed = benchmarks["GW170817_GRB170817A_SPEED"]["value"]
    mass_limit = benchmarks["GWTC3_IMPROVED_DISPERSION_2026"]["value"][
        "graviton_mass_90pct_upper_eV_c2"
    ]
    records: list[dict[str, Any]] = []
    for branch in config["branch_catalog"]:
        universal = branch["interpretation"].startswith("UNIVERSAL_RADIATIVE")
        speed_pass: bool | None = None
        mass_pass: bool | None = None
        if universal:
            speed_pass = (
                speed["delta_c_over_c_min"]
                <= branch["delta_c_over_c"]
                <= speed["delta_c_over_c_max"]
            )
            mass_pass = branch["graviton_mass_eV_c2"] <= mass_limit
        conservation_pass = branch["Gamma"] == 0.0
        finite_front_claim = branch["zeta"] == 0.0
        if not universal:
            decision = "RETAINED_NONUNIVERSAL_BRANCH_NEEDS_COUPLING_AND_SCREENING_COMPLETION"
        elif not speed_pass:
            decision = "FALSIFIED_IF_UNIVERSAL_RADIATIVE_BY_GW170817_SPEED"
        elif not mass_pass:
            decision = "FALSIFIED_IF_TESTED_TENSOR_MASS_BY_2026_GWTC3_BOUND"
        elif not conservation_pass:
            decision = "RETAINED_OPEN_SYSTEM_BRANCH_BATH_REQUIRED"
        elif not finite_front_claim:
            decision = "RETAINED_EFT_BRANCH_CUTOFF_REQUIRED_NO_FINITE_FRONT_CLAIM"
        else:
            decision = "PASSES_FROZEN_PROPAGATION_BENCHMARKS_NOT_A_THEORY_PASS"
        records.append(
            {
                **branch,
                "universal_radiative_benchmarks_applicable": universal,
                "gw170817_speed_pass": speed_pass,
                "tensor_mass_bound_pass": mass_pass,
                "closed_field_energy_conservation_pass": conservation_pass,
                "finite_front_claim_available": finite_front_claim,
                "solar_static_limit": "PASS_EXACT_POISSON",
                "solar_moving_body": "NOT_DERIVED_COVARIANT_COMPLETION_REQUIRED",
                "binary_radiation_reaction": "NOT_DERIVED_SOURCE_GENERATION_REQUIRED",
                "decision": decision,
            }
        )
    return records


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    integrity = _validate_local_integrity(base)
    analytic = analytic_controls()
    recoveries = injection_recovery(config)
    branches = classify_branches(config)
    decision_counts: dict[str, int] = {}
    for row in branches:
        decision_counts[row["decision"]] = decision_counts.get(row["decision"], 0) + 1
    checks = {
        "T01_LOCAL_BYTES_SEALED": len(integrity) == 4,
        "T02_CAUSAL_LAW_AND_COMMON_CLOCK_FROZEN": True,
        "T03_STATIC_SPEED_ONLY_NO_ENHANCEMENT_EXACT": analytic[
            "A01_STATIC_POISSON_INDEPENDENT_OF_CG"
        ]["passed"],
        "T04_DAMPING_DROPS_FROM_STATIC_LIMIT": analytic["A02_STATIC_DAMPING_DROPS_OUT"]["passed"],
        "T05_GR_KERNEL_CONTROL": analytic["A03_LUMINAL_MASSLESS_CONTROL"]["passed"],
        "T06_CONSERVATIVE_ENERGY_IDENTITY": analytic["A04_CONSERVATIVE_ENERGY_IDENTITY"]["passed"],
        "T07_ATTENUATION_ENERGY_SINK_EXPOSED": analytic["A06_DAMPING_REQUIRES_ENERGY_SINK"][
            "passed"
        ],
        "T08_RETARDED_POLES_FACTORIZED": analytic["A07_RETARDED_POLE_FACTORIZATION"]["passed"],
        "T09_QUARTIC_FRONT_LIMIT_EXPOSED": analytic["A08_QUARTIC_PHASE_SPEED_UNBOUNDED"]["passed"],
        "T10_TARGET_FREE_INJECTIONS_RECOVERED": all(row["exact_recovery"] for row in recoveries),
        "T11_UNCONVENTIONAL_BRANCHES_RETAINED": any(
            row["decision"].startswith("RETAINED") for row in branches
        ),
        "T12_GW_SPEED_AND_MASS_FALSIFIERS_EXECUTED": sum(
            row["decision"].startswith("FALSIFIED") for row in branches
        )
        == 3,
        "T13_SOLAR_AND_BINARY_APPLICABILITY_GATES": all(
            row["solar_moving_body"].startswith("NOT_DERIVED")
            and row["binary_radiation_reaction"].startswith("NOT_DERIVED")
            for row in branches
        ),
        "T14_PUBLIC_DATA_PRODUCT_EXACTLY_SPECIFIED": config["public_data_preflight"]["dataset_doi"]
        == "10.7935/K5B8566F",
        "T15_NO_OBSERVATIONAL_OR_PAID_ACCESS": all(
            value == 0 for value in config["access_ledger"].values()
        ),
        "T16_CLAIM_CEILING_RETAINED": config["claim_boundary"]["publication_ready"] is False,
    }
    _require(all(checks.values()), "receipt checks failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "target_free_differential_propagation_kernel_derived_real_data_not_scored",
        "decision": DECISION,
        "law": config["law"],
        "assumptions": config["assumptions"],
        "derived_observables": config["derived_observables"],
        "analytic_controls": analytic,
        "target_free_recovery": recoveries,
        "branch_adjudication": branches,
        "branch_decision_counts": decision_counts,
        "published_benchmarks": config["frozen_published_benchmarks"],
        "solar_and_binary_constraint_boundary": {
            "solar_static": "Exact Poisson recovery; no speed-only enhancement.",
            "solar_dynamic": "No aberration or PPN claim until a covariant completion supplies all cancellation terms.",
            "binary": "B1913+16 and the Double Pulsar are frozen mandatory controls, but this propagation-only packet has no emission or radiation-reaction prediction to score.",
        },
        "strongest_counterexample": {
            "result": "A gravity speed different from light does not by itself increase the force of a stationary source.",
            "proof": "At omega=0 with mu=zeta=0, the c_g^2 source normalization cancels the c_g^2 spatial operator exactly, leaving Phi/rho=-4 pi G/k^2.",
            "consequence": "Galaxy and relaxed-cluster anomalies require source history, nonzero temporal frequency, mass/dispersion, attenuation with a physical reservoir, nonlinear response, or a distinct screened sector; a bare propagation-speed dial cannot explain them.",
        },
        "novelty_assessment": {
            "neighbors": config["novelty_neighbors"],
            "known_components": "Modified speed, mass, dispersion, and GW friction are established research families.",
            "candidate_unique_unit": "The potentially publishable unit is the joint response-free audit: one causal source-to-force kernel, its exact static no-enhancement theorem, explicit conservation/front-speed applicability gates, target-free recovery, and a branch-by-branch mapping to current GW, Solar, and binary controls.",
            "historical_novelty_established": False,
            "publication_status": "METHODS_OR_NO_GO_CANDIDATE_REQUIRES_EXACT_PRIOR_ART_SEARCH_AND_REAL_DATA_CONTROL",
        },
        "public_data_preflight": config["public_data_preflight"],
        "next_empirical_run": config["public_data_preflight"]["next_empirical_run"],
        "claim_boundary": config["claim_boundary"],
        "access_ledger": config["access_ledger"],
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "integrity": integrity,
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("decision") == DECISION, "receipt decision changed")
    _require(
        receipt.get("checks_passed") == receipt.get("checks_total") == 16,
        "checks failed",
    )
    _require(all(receipt.get("checks", {}).values()), "receipt contains failed check")
    _require(receipt.get("claim_boundary") == config["claim_boundary"], "claims changed")
    _require(receipt.get("access_ledger") == config["access_ledger"], "access changed")
    _require(
        receipt.get("strongest_counterexample", {}).get("result")
        == "A gravity speed different from light does not by itself increase the force of a stationary source.",
        "counterexample changed",
    )
    _require(receipt.get("content_sha256") == _self_hash(receipt), "content hash invalid")


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return "EXISTING_IDENTICAL"
        raise DifferentialPropagationError(f"refusing to replace nonidentical artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return "CREATED"


def _command_build(root: Path, output: Path) -> int:
    receipt = build_receipt(root)
    validate_receipt(receipt, load_config(root))
    state = _atomic_no_clobber(output, receipt)
    print(json.dumps({"state": state, "output": str(output), "decision": DECISION}))
    return 0


def _command_check(root: Path, output: Path) -> int:
    config = load_config(root)
    receipt = _read_json(output)
    validate_receipt(receipt, config)
    expected = build_receipt(root)
    _require(receipt == expected, "receipt is not deterministic")
    print(json.dumps({"state": "VALID", "output": str(output), "decision": DECISION}))
    return 0


def _command_status(root: Path, output: Path) -> int:
    config = load_config(root)
    receipt = _read_json(output)
    validate_receipt(receipt, config)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "decision": receipt["decision"],
                "publication_ready": receipt["claim_boundary"]["publication_ready"],
                "real_observational_rows_scored": receipt["claim_boundary"][
                    "real_observational_rows_scored"
                ],
                "next_empirical_run": receipt["next_empirical_run"],
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    parser.add_argument("--root", type=Path, default=_repo_root())
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    output = arguments.output or (root / OUTPUT_PATH)
    if not output.is_absolute():
        output = root / output
    if arguments.command == "build":
        return _command_build(root, output)
    if arguments.command == "check":
        return _command_check(root, output)
    return _command_status(root, output)


if __name__ == "__main__":
    raise SystemExit(main())
