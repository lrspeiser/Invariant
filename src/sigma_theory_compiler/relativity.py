from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy as sp

G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_KG = 1.98847e30
R_SUN_M = 6.957e8
AU_M = 149_597_870_700.0
KPC_M = 3.085677581491367e19
JULIAN_YEAR_DAYS = 365.25


def _result(name: str, status: str, claim: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": status, "claim": claim, "evidence": evidence}


@lru_cache(maxsize=1)
def schwarzschild_ricci_components() -> dict[str, str]:
    """Calculate R_ab directly from the Schwarzschild metric.

    This does not use a stored list of known-zero components. Christoffel symbols
    and the Ricci tensor are constructed from the metric definition.
    """

    time, radius, theta, phi = sp.symbols("t r theta phi", real=True)
    schwarzschild_radius = sp.symbols("r_s", positive=True)
    coordinates = (time, radius, theta, phi)
    lapse = 1 - schwarzschild_radius / radius
    metric = sp.diag(-lapse, 1 / lapse, radius**2, radius**2 * sp.sin(theta) ** 2)
    inverse = sp.simplify(metric.inv())
    size = 4

    christoffel = [[[sp.Integer(0) for _ in range(size)] for _ in range(size)] for _ in range(size)]
    for upper in range(size):
        for left in range(size):
            for right in range(size):
                value = sp.Integer(0)
                for contracted in range(size):
                    value += inverse[upper, contracted] * (
                        sp.diff(metric[contracted, right], coordinates[left])
                        + sp.diff(metric[contracted, left], coordinates[right])
                        - sp.diff(metric[left, right], coordinates[contracted])
                    )
                christoffel[upper][left][right] = sp.simplify(value / 2)

    components: dict[str, str] = {}
    names = ("t", "r", "theta", "phi")
    for left in range(size):
        for right in range(size):
            value = sp.Integer(0)
            for contracted in range(size):
                value += sp.diff(
                    christoffel[contracted][left][right], coordinates[contracted]
                )
                value -= sp.diff(
                    christoffel[contracted][left][contracted], coordinates[right]
                )
                for inner in range(size):
                    value += (
                        christoffel[contracted][left][right]
                        * christoffel[inner][contracted][inner]
                    )
                    value -= (
                        christoffel[inner][left][contracted]
                        * christoffel[contracted][right][inner]
                    )
            components[f"R_{names[left]}{names[right]}"] = str(sp.trigsimp(sp.simplify(value)))
    return components


def schwarzschild_vacuum_check() -> dict[str, Any]:
    components = schwarzschild_ricci_components()
    nonzero = {name: value for name, value in components.items() if value != "0"}
    return _result(
        "schwarzschild_vacuum",
        "pass" if not nonzero else "fail",
        "The Schwarzschild exterior metric has R_ab = 0 when calculated from its metric components.",
        {
            "metric": "diag(-(1-r_s/r), (1-r_s/r)^-1, r^2, r^2 sin(theta)^2)",
            "components_calculated": len(components),
            "nonzero_components": nonzero,
            "all_components": components,
            "domain": "r > r_s, excluding the coordinate singularity",
        },
    )


def ppn_reference_check() -> dict[str, Any]:
    u = sp.symbols("u", real=True)
    g_tt = -((1 - u / 2) / (1 + u / 2)) ** 2
    spatial_factor = (1 + u / 2) ** 4
    g_tt_series = sp.series(g_tt, u, 0, 3).removeO().expand()
    spatial_series = sp.series(spatial_factor, u, 0, 3).removeO().expand()
    beta = sp.simplify(-g_tt_series.coeff(u, 2) / 2)
    gamma = sp.simplify(spatial_series.coeff(u, 1) / 2)
    passed = beta == 1 and gamma == 1
    return _result(
        "gr_ppn_recovery",
        "pass" if passed else "fail",
        "The isotropic Schwarzschild expansion recovers PPN beta = gamma = 1.",
        {
            "u_definition": "G M/(rho c^2)",
            "g_tt_exact": str(g_tt),
            "spatial_factor_exact": str(spatial_factor),
            "g_tt_series": str(g_tt_series),
            "spatial_factor_series": str(spatial_series),
            "beta": str(beta),
            "gamma": str(gamma),
            "cassini_gamma_minus_one": float(gamma - 1),
            "frozen_cassini_absolute_bound": 2.3e-5,
        },
    )


def solar_system_numeric_checks() -> list[dict[str, Any]]:
    solar_mu = G_SI * M_SUN_KG

    mercury_a_m = 57.90905e9
    mercury_eccentricity = 0.205630
    mercury_period_days = 87.9691
    advance_per_orbit_rad = (
        6
        * math.pi
        * solar_mu
        / (mercury_a_m * (1 - mercury_eccentricity**2) * C_SI**2)
    )
    orbits_per_century = 100 * JULIAN_YEAR_DAYS / mercury_period_days
    advance_arcsec_century = advance_per_orbit_rad * orbits_per_century * 206_264.806247
    mercury_pass = 42.5 <= advance_arcsec_century <= 43.5

    deflection_rad = 4 * solar_mu / (R_SUN_M * C_SI**2)
    deflection_arcsec = deflection_rad * 206_264.806247
    deflection_pass = 1.74 <= deflection_arcsec <= 1.76

    # Leading one-way superior-conjunction delay relative to flat propagation.
    # Earth-to-Mercury with a solar-limb impact parameter is a frozen geometry,
    # not a comparison to a particular mission data reduction.
    mercury_distance_m = 0.387098 * AU_M
    shapiro_seconds = 2 * solar_mu / C_SI**3 * math.log(
        4 * AU_M * mercury_distance_m / R_SUN_M**2
    )
    shapiro_pass = 0.00010 <= shapiro_seconds <= 0.00013

    return [
        _result(
            "mercury_perihelion",
            "pass" if mercury_pass else "fail",
            "GR recovers approximately 43 arcseconds per century of anomalous Mercury perihelion advance.",
            {
                "calculated_arcsec_per_century": advance_arcsec_century,
                "acceptance_interval": [42.5, 43.5],
                "formula": "6*pi*G*M/(a*(1-e^2)*c^2) per orbit",
            },
        ),
        _result(
            "solar_limb_light_deflection",
            "pass" if deflection_pass else "fail",
            "GR recovers approximately 1.75 arcseconds of solar-limb light deflection.",
            {
                "calculated_arcsec": deflection_arcsec,
                "acceptance_interval": [1.74, 1.76],
                "formula": "4*G*M/(R_sun*c^2)",
            },
        ),
        _result(
            "shapiro_delay_geometry_control",
            "pass" if shapiro_pass else "fail",
            "GR produces the expected order of magnitude for a frozen Earth-Mercury superior-conjunction Shapiro delay geometry.",
            {
                "calculated_one_way_microseconds": shapiro_seconds * 1e6,
                "acceptance_interval_microseconds": [100.0, 130.0],
                "warning": "Geometry control only; not a fit to spacecraft tracking residuals.",
            },
        ),
    ]


def galaxy_exterior_control() -> dict[str, Any]:
    # For two points in the exterior of the same finite baryonic source,
    # r_2/r_1 = theta_2/theta_1. Distance and absolute mass therefore cancel:
    # v_2/v_1 = sqrt(theta_1/theta_2).
    angular_radius_ratios = [1.0, 2.0, 4.0]
    predicted_speed_ratios = [1 / math.sqrt(ratio) for ratio in angular_radius_ratios]
    predicted_ratio = predicted_speed_ratios[-1]
    flat_ratio = 1.0
    mismatch = abs(predicted_ratio - flat_ratio) > 0.25
    return _result(
        "asymptotic_baryons_only_galaxy",
        "expected_mismatch" if mismatch else "unexpected_agreement",
        "Outside a finite baryonic mass, weak-field GR predicts v_c proportional to r^-1/2, not a flat rotation curve.",
        {
            "hypothesis_tested": "GR sourced only by a frozen finite baryonic mass",
            "excluded_rescues": [
                "unobserved source components",
                "halo parameters inferred from the same rotation curve",
                "redshift-derived distance"
            ],
            "angular_radius_ratios": angular_radius_ratios,
            "predicted_speed_ratios": predicted_speed_ratios,
            "v_at_4theta_over_v_at_theta": predicted_ratio,
            "flat_curve_ratio": flat_ratio,
            "predicted_log_slope": -0.5,
            "flat_log_slope": 0.0,
            "interpretation": (
                "The distance and absolute baryonic mass cancel from this exterior ratio test. "
                "A future data run can use angular radii and Doppler-velocity ratios directly."
            ),
            "warning": (
                "This is a theoretical asymptotic control, not an observational likelihood. A real "
                "source must first demonstrate that both points lie outside the measured baryonic extent."
            ),
        },
    )


def run_relativity_reference_suite(
    formal_eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_checks = [schwarzschild_vacuum_check(), ppn_reference_check()]
    raw_checks.extend(solar_system_numeric_checks())
    eligibility_pass = (
        formal_eligibility is not None
        and formal_eligibility.get("status") == "eligible"
        and formal_eligibility.get("reference_controls_allowed") is True
    )
    checks = raw_checks
    if not eligibility_pass:
        checks = [
            {
                **check,
                "status": "blocked",
                "evidence": {
                    **check["evidence"],
                    "formal_prerequisite": "not eligible",
                },
            }
            for check in raw_checks
        ]
    galaxy = galaxy_exterior_control()
    passed = sum(check["status"] == "pass" for check in checks)
    failed = sum(check["status"] == "fail" for check in checks)
    blocked = sum(check["status"] == "blocked" for check in checks)
    return {
        "schema_version": "sigma-relativity-reference-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "formal_prerequisite": formal_eligibility or {
            "status": "ineligible",
            "errors": ["no action-health eligibility certificate supplied"],
            "reference_controls_allowed": False,
            "observational_dataset_opened": False,
        },
        "reference_action": {
            "name": "Einstein-Hilbert",
            "action": "S = integral sqrt(-g) [c^3 R/(16 pi G) + L_m/c] d^4x",
            "expected_field_equation": "G_ab = 8 pi G T_ab/c^4",
            "action_variation_engine_status": "pass" if eligibility_pass else "blocked",
            "constraint_and_degree_count_status": "pass" if eligibility_pass else "blocked",
            "physical_hamiltonian_status": "pass" if eligibility_pass else "blocked",
            "physical_principal_symbol_status": "pass" if eligibility_pass else "blocked",
        },
        "counts": {
            "golden_total": len(checks),
            "passed": passed,
            "failed": failed,
            "blocked": blocked,
        },
        "golden_checks": checks,
        "galaxy_negative_control": galaxy,
        "interpretation": (
            "The golden checks validate the GR reference solver only when bound to a passing "
            "Einstein-Hilbert formal-health chain. The galaxy control demonstrates an expected "
            "mismatch for GR plus the measured finite baryonic source only. Unobserved source "
            "components are outside this project's evidence policy and cannot be used as a rescue."
        ),
        "observational_evidence_policy": {
            "allowed": [
                "raw detector counts and calibrated spectra",
                "angular positions and angular-radius ratios",
                "proper motions and time delays",
                "Doppler wavelength and velocity ratios with calibration provenance",
                "measured baryonic tracers with explicit uncertainty and transformations"
            ],
            "redshift_rule": (
                "A spectral wavelength shift may be used as a measured dimensionless ratio. It may "
                "not be converted to distance, mass, or environment without a separately authorized "
                "and audited distance inference."
            ),
            "excluded": [
                "dark-matter or invisible-halo quantities as observations or target labels",
                "halo fits as a rescue for a failed baryons-only prediction",
                "redshift-derived cosmological distances by default",
                "supernova distance moduli by default"
            ]
        },
        "deferred": [
            "candidate-specific background solution and static dictionary",
            "an independently audited direct-observation dataset manifest",
            "measured extended-source likelihood using only policy-permitted observables",
        ],
    }


def write_relativity_report(report: dict[str, Any], output_directory: str | Path) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "relativity_reference.json"
    markdown_path = output / "relativity_reference.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    galaxy = report["galaxy_negative_control"]
    lines = [
        "# GR, Solar-System, and galaxy-exterior reference run",
        "",
        f"Golden GR checks: {report['counts']['passed']}/{report['counts']['golden_total']} passed.",
        "",
        "| Check | Status | Key result |",
        "|---|---|---|",
    ]
    for check in report["golden_checks"]:
        evidence = check["evidence"]
        key = (
            evidence.get("calculated_arcsec_per_century")
            or evidence.get("calculated_arcsec")
            or evidence.get("calculated_one_way_microseconds")
            or evidence.get("beta")
            or f"{evidence.get('components_calculated', '')} Ricci components"
        )
        lines.append(f"| {check['name']} | {check['status']} | {key} |")
    lines.extend(
        [
            "",
            "## Galaxy exterior control",
            "",
            f"Status: **{galaxy['status']}**.",
            "",
            galaxy["claim"],
            "",
            "At angular-radius ratios 1:2:4, the baryons-only exterior prediction gives velocity ratios "
            + ":".join(f"{value:.3f}" for value in galaxy["evidence"]["predicted_speed_ratios"])
            + ". Distance and absolute mass cancel.",
            "",
            "This is a failure of the **GR + measured-baryons-only hypothesis** to produce a flat asymptotic curve. Unobserved source components are not an allowed rescue in this project.",
            "",
            "## Still deferred",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["deferred"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
